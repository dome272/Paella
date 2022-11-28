import torch
import torch.nn as nn
import numpy as np
from fast_pytorch_kmeans import KMeans
from torchtools.nn import VectorQuantize

BASE_SHAPE = (6, 16, 16)


class ResBlockvq(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, scaler=None, kernel_size=3):
        super().__init__()
        self.resblock = nn.Sequential(
            nn.GELU(),
            nn.Conv3d((c + c_cond), c_hidden, kernel_size=1),
            nn.GELU(),
            nn.ReplicationPad3d(kernel_size // 2),
            nn.Conv3d(c_hidden, c_hidden, kernel_size=kernel_size, groups=c_hidden),
            nn.GELU(),
            nn.Conv3d(c_hidden, c, kernel_size=1),
        )
        self.scaler = scaler

    def forward(self, x, s=None, encoder=True, i=None):
        res = x
        if s is not None:
            x = torch.cat([x, s], dim=1)
        x = res + self.resblock(x)
        if self.scaler is not None:
            if encoder:
                x = x.permute(0, 2, 1, 3, 4)
                x = self.scaler(x)
                x = x.permute(0, 2, 1, 3, 4)
            else:
                x = self.scaler(x)
                if i == 1:
                    x = x[:, :, 1:]
        return x


class Encoder(nn.Module):
    def __init__(self, c_in, c_hidden=256, levels=4, blocks_per_level=1, c_min=4, bottleneck_blocks=8):
        super().__init__()
        levels = levels - 2
        c_first = max(c_hidden // (4 ** max(levels - 1, 0)), c_min)
        self.stem = nn.Sequential(
            nn.Conv3d(c_in, c_first, kernel_size=(2, 4, 4), stride=(2, 4, 4)),
        )
        self.encoder = nn.ModuleList()
        self.remapper = nn.ModuleList()
        for i in range(levels):
            for j in range(blocks_per_level):
                bc_in_raw = c_hidden // (4 ** (levels - i - 1))
                bc_in_next_raw = c_hidden // (4 ** (levels - i))
                bc_in = max(bc_in_raw, c_min)
                bc_in_next = max(bc_in_next_raw, c_min)
                bc_hiden = bc_in * 4
                bc_cond = 0 if i == 0 and j == 0 else bc_in if j > 0 else bc_in_next
                scaler = nn.PixelUnshuffle(2) if i < (levels - 1) and j == (blocks_per_level - 1) else None
                if scaler is not None and bc_in_raw < bc_in:
                    self.remapper.append(nn.Sequential(
                        nn.Conv3d(bc_in * 4, bc_in, kernel_size=1),
                    ))
                else:
                    self.remapper.append(nn.Identity())
                self.encoder.append(ResBlockvq(bc_in, bc_hiden, c_cond=bc_cond, scaler=scaler))
        for block in self.encoder:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))

        self.bottleneck = nn.Sequential(*[ResBlockvq(c_hidden, c_hidden * 4) for _ in range(bottleneck_blocks)])
        for block in self.bottleneck:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))

        self.learned_frame = nn.Parameter(torch.randn(3, 1, 128, 128) / (c_hidden ** 0.5))

    def forward(self, image, video=None):
        image = torch.cat([self.learned_frame.unsqueeze(0).expand(image.shape[0], -1, -1, -1, -1), image.unsqueeze(2)], dim=2)
        if video is not None:
            video = video.permute(0, 2, 1, 3, 4)
            video = torch.cat([image, video], dim=2)
        else:
            video = image
        video = self.stem(video)
        s = None
        if len(self.encoder) > 0:
            for block, remapper in zip(self.encoder, self.remapper):
                if block.scaler is not None:
                    prev_s = nn.functional.interpolate(video, scale_factor=(1, 0.5, 0.5), recompute_scale_factor=False)
                else:
                    prev_s = video
                video = block(video, s)
                if block.scaler is not None:
                    video = remapper(video)
                s = prev_s
        video = self.bottleneck(video)
        return video


class Decoder(nn.Module):
    def __init__(self, c_out, c_hidden=256, levels=4, blocks_per_level=2, bottleneck_blocks=8, c_min=4, ucm=4,
                 out_ks=1):
        super().__init__()
        self.bottleneck = nn.Sequential(*[ResBlockvq(c_hidden, c_hidden * 4) for _ in range(bottleneck_blocks)])
        for block in self.bottleneck:
            block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))

        self.decoder = nn.ModuleList()
        self.blocks_per_level = blocks_per_level
        for i in range(levels):
            for j in range(blocks_per_level):
                bc_in_raw = c_hidden // (ucm ** i)
                bc_in_prev_raw = c_hidden // (ucm ** (i - 1))
                bc_in = max(bc_in_raw, c_min)
                bc_in_prev = max(bc_in_prev_raw, c_min)
                bc_hiden = bc_in * 4
                bc_cond = 0 if i == 0 and j == 0 else bc_in if j > 0 else bc_in_prev
                if i < (levels - 1) and j == (blocks_per_level - 1):
                    if i == 0:
                        scaler = nn.Sequential(
                            nn.Upsample(scale_factor=(2, 2, 2)),
                            nn.Conv3d(bc_in, max(bc_in // ucm, c_min), kernel_size=1),
                        )
                    else:
                        scaler = nn.Sequential(
                            nn.Upsample(scale_factor=(1, 2, 2)),
                            nn.Conv3d(bc_in, max(bc_in // ucm, c_min), kernel_size=1),
                        )
                else:
                    scaler = None
                block = ResBlockvq(bc_in, bc_hiden, c_cond=bc_cond, scaler=scaler)
                block.resblock[-1].weight.data *= np.sqrt(1 / (levels * blocks_per_level + bottleneck_blocks))
                self.decoder.append(block)
        self.output_convs = nn.ModuleList()
        for i in range(levels):
            bc_in = max(c_hidden // (ucm ** min(i + 1, levels - 1)), c_min)
            self.output_convs.append(nn.Sequential(
                nn.ReflectionPad3d(out_ks // 2),
                nn.Conv3d(bc_in, c_out, kernel_size=out_ks)
            ))  # kernel_size=7, padding=3 <-- NO FUNCTIONA

    def forward(self, x):
        x = self.bottleneck(x)
        s = None
        outs = []
        for i, block in enumerate(self.decoder):
            if block.scaler is not None:
                if i == 1:
                    prev_s = nn.functional.interpolate(x, scale_factor=(2, 2, 2), recompute_scale_factor=False)[:, :, 1:]
                else:
                    prev_s = nn.functional.interpolate(x, scale_factor=(1, 2, 2), recompute_scale_factor=False)
            else:
                prev_s = x
            x = block(x, s, encoder=False, i=i)
            s = prev_s
            if block.scaler is not None or i == len(self.decoder) - 1:
                outs.append(self.output_convs[i // self.blocks_per_level](x))
        for i in range(len(outs) - 1):
            if outs[i].size(3) < outs[-1].size(3):
                outs[i] = nn.functional.interpolate(outs[i], size=(outs[-1].size(2), outs[-1].size(3), outs[-1].size(4)), mode='nearest')
        x = torch.stack(outs, dim=0).sum(0)
        return x.sigmoid().permute(0, 2, 1, 3, 4)


class VQModule(nn.Module):
    def __init__(self, c_hidden, k, q_init, q_refresh_step, q_refresh_end, reservoir_size=int(9e4)):
        super().__init__()
        self.vquantizer = VectorQuantize(c_hidden, k=k, ema_loss=True)
        self.codebook_size = k
        self.q_init, self.q_refresh_step, self.q_refresh_end = q_init, q_refresh_step, q_refresh_end
        self.register_buffer('q_step_counter', torch.tensor(0))
        self.reservoir = None
        self.reservoir_size = reservoir_size

    def forward(self, x, dim=-1):
        if self.training:
            # self.q_step_counter += x.size(0)
            # x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            # self.reservoir = x_flat if self.reservoir is None else torch.cat([self.reservoir, x_flat], dim=0)
            # self.reservoir = self.reservoir[torch.randperm(self.reservoir.size(0))[:self.reservoir_size]].detach()
            # if self.q_step_counter < self.q_init:
            #     qe, commit_loss, indices = x, x.new_tensor(0), None
            # else:
                # if self.q_step_counter < self.q_init + self.q_refresh_end:
                #     if (self.q_step_counter + self.q_init) % self.q_refresh_step == 0 or self.q_step_counter == self.q_init or self.q_step_counter == self.q_init + self.q_refresh_end - 1:
                #         print("Running KMeans")
                #         kmeans = KMeans(n_clusters=self.codebook_size, mode='euclidean', verbose=0)
                #         kmeans.fit_predict(self.reservoir)
                #         self.vquantizer.codebook.weight.data = kmeans.centroids.detach()
            qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)
        else:
            if self.q_step_counter < self.q_init:
                qe, commit_loss, indices = x, x.new_tensor(0), None
            else:
                qe, (_, commit_loss), indices = self.vquantizer(x, dim=dim)

        return qe, commit_loss, indices


class VIVQ(nn.Module):
    def __init__(self, base_channels=3, c_hidden=512, c_codebook=16, codebook_size=1024):
        super().__init__()
        self.encoder = Encoder(base_channels, c_hidden=c_hidden)
        self.cod_mapper = nn.Sequential(
            nn.Conv3d(c_hidden, c_codebook, kernel_size=1),
            nn.BatchNorm3d(c_codebook),
        )
        self.cod_unmapper = nn.Conv3d(c_codebook, c_hidden, kernel_size=1)
        self.decoder = Decoder(base_channels, c_hidden=c_hidden)

        self.codebook_size = codebook_size
        self.vqmodule = VQModule(
            c_codebook, k=codebook_size,
            q_init=1000, q_refresh_step=1000, q_refresh_end=5000
            # q_init=15010 * 20, q_refresh_step=15010, q_refresh_end=15010 * 130
        )

    def encode(self, image, video):
        x = self.encoder(image, video)  # B x T x (H x W) x C
        x = self.cod_mapper(x)
        shape = x.shape
        x = x.view(*x.shape[:3], x.shape[3]*x.shape[4]).permute(0, 2, 3, 1)
        qe, commit_loss, indices = self.vqmodule(x, dim=-1)
        # indices = indices.view(image.shape[0], -1)
        if video is not None:
            indices = indices.view(image.shape[0], *BASE_SHAPE)
        else:
            indices = indices.view(image.shape[0], *BASE_SHAPE[1:]).unsqueeze(1)
        return (x, qe), commit_loss, indices, shape

    def decode(self, x, shape=None):
        if shape is not None:
            x = x.permute(0, 3, 1, 2).view(shape)
        x = self.cod_unmapper(x)
        x = self.decoder(x)
        return x

    def decode_indices(self, x, shape=None):
        if shape is not None:
            x = x.view(x.shape[0], *shape)
        return self.decode(self.vqmodule.vquantizer.idx2vq(x, dim=-1).permute(0, 4, 1, 2, 3))

    def forward(self, image, video=None):
        # print(image.shape, video.shape)
        (_, qe), commit_loss, _, shape = self.encode(image, video)
        # print(qe.shape)
        decoded = self.decode(qe, shape)
        # print(decoded.shape)
        return decoded, commit_loss


if __name__ == '__main__':
    device = "cuda"
    image = torch.randn(1, 3, 128, 128).to(device)
    video = torch.randn(1, 10, 3, 128, 128).to(device)
    video = None
    # e = Encoder(c_in=3).to(device)
    # d = Decoder(c_out=3).to(device)
    vq = VIVQ(c_hidden=512).to(device)
    print(sum([p.numel() for p in vq.parameters()]))
    # rb = ResBlockvq(3, 100).to(device)
    # print(rb(x).shape)
    # r = e(image, video)
    # print(r.shape)
    # print(video.shape)
    # print(d(r).shape)
    vq(image, video)
