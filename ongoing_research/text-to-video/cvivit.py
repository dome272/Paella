import torch
import torch.nn as nn
from torchtools.nn import VectorQuantize
from fast_pytorch_kmeans import KMeans


class TemporalSpatialAttention(nn.Module):
    def __init__(self, channels, size, frames, num_layers=4, num_heads=4, spatial_first=True, pos_encodings=True):
        super(TemporalSpatialAttention, self).__init__()
        self.spatial_first = spatial_first
        self.pos_encodings = pos_encodings

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels, dim_feedforward=channels * 4,
                                                            nhead=num_heads, batch_first=True, norm_first=True,
                                                            activation='gelu')
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers,
                                                         norm=nn.LayerNorm(channels))
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer, num_layers=num_layers,
                                                          norm=nn.LayerNorm(channels))

        if pos_encodings:
            self.spatial_positional_encoding = nn.Parameter(torch.randn(1, 1, size * size, channels) / channels ** 0.5)
            self.temporal_positional_encoding = nn.Parameter(torch.randn(1, frames, 1, channels) / channels ** 0.5)

    def _spatial_attn(self, x, base_shape):
        x = x.view(-1, *x.shape[2:])  # B x T x (H x W) x C -> (B x T) x (H x W) x C
        x = self.spatial_transformer(x)
        x = x.view(base_shape[0], base_shape[1], *x.shape[1:])  # (B x T) x (H x W) x C -> B x T x (H x W) x C
        return x

    def _temporal_attn(self, x, base_shape):
        mask = torch.triu(torch.ones(base_shape[1], base_shape[1]) * float('-inf'), diagonal=1).to(x.device)
        x = x.permute(0, 2, 1, 3).view(-1, x.size(1), x.size(-1))  # B x T x (H x W) x C -> (B x H x W) x T x C
        x = self.temporal_transformer(x, mask=mask)
        x = x.view(base_shape[0], -1, *x.shape[1:]).permute(0, 2, 1, 3)  # (B x H x W) x T x C -> B x T x (H x W) x C
        return x

    def forward(self, x):
        base_shape = x.shape  # x -> B x T x (H x W) x C
        # if self.pos_encodings:
        #     x = x + self.spatial_positional_encoding + self.temporal_positional_encoding[:, :x.shape[1]]
        if self.spatial_first:
            x += self.spatial_positional_encoding
            x = self._spatial_attn(x, base_shape)
            x += self.temporal_positional_encoding
            x = self._temporal_attn(x, base_shape)
        else:
            x += self.temporal_positional_encoding
            x = self._temporal_attn(x, base_shape)
            x += self.spatial_positional_encoding
            x = self._spatial_attn(x, base_shape)
        return x


class Encoder(nn.Module):
    def __init__(self, patch_size=(5, 8, 8), input_channels=3, hidden_channels=64, size=32, compressed_frames=20,
                 num_layers=4, num_heads=4):
        super(Encoder, self).__init__()
        self.video_patch_emb = nn.Conv3d(input_channels, hidden_channels, kernel_size=patch_size, stride=patch_size)
        self.image_patch_emb = nn.Conv2d(input_channels, hidden_channels, kernel_size=patch_size[1:], stride=patch_size[1:])
        self.attention = TemporalSpatialAttention(hidden_channels, size, compressed_frames+1, num_layers=num_layers,
                                                  num_heads=num_heads)

    def forward(self, image, video):
        # image, video: 1 x 3 x 128 x 128, 1 x 100 x 3 x 128 x 128
        image = self.image_patch_emb(image)  # 1 x 64 x 16 x 16
        if video is not None:
            video = video.permute(0, 2, 1, 3, 4)  # B x T x C x H x W -> B x C x T x H x W
            video = self.video_patch_emb(video)  # 1 x 64 x 20 x 16 x 16
            video = torch.cat([image.unsqueeze(2), video], dim=2)  # 1 x 64 x 21 x 16 x 16
        else:
            video = image.unsqueeze(2)
        video = video.view(*video.shape[:3], -1).permute(0, 2, 3, 1)  # B x T x (H x W) x C -> 1 x 21 x (16*16) x 64
        video = self.attention(video)  # 1 x 21 x 256 x 64
        return video


class Decoder(nn.Module):
    def __init__(self, patch_size=(5, 8, 8), input_channels=3, hidden_channels=64, size=32, compressed_frames=20,
                 num_layers=4, num_heads=4):
        super(Decoder, self).__init__()
        self.size = size
        self.attention = TemporalSpatialAttention(hidden_channels, size, compressed_frames+1, num_layers=num_layers,
                                                  num_heads=num_heads, spatial_first=False)
        self.video_unpatch_emb = nn.ConvTranspose3d(hidden_channels, input_channels, kernel_size=patch_size, stride=patch_size)
        self.image_unpatch_emb = nn.ConvTranspose2d(hidden_channels, input_channels, kernel_size=patch_size[1:], stride=patch_size[1:])

    def forward(self, x):
        # example x: 1 x 21 x 256 x 64
        x = self.attention(x)  # 1 x 21 x 256 x 64
        x = x.permute(0, 3, 1, 2).view(x.size(0), x.size(3), x.size(1), self.size, self.size)  # B x T x (H x W) x C -> B x C x T x H x W
        if x.shape[2] > 1:  # not only image training
            image, video = x[:, :, 0], x[:, :, 1:]
            video = self.video_unpatch_emb(video)
            image = self.image_unpatch_emb(image)
            x = torch.cat([image.unsqueeze(2), video], dim=2).permute(0, 2, 1, 3, 4)  # B x C x T x H x W -> B x T x C x H x W
        else:
            image = x[:, :, 0]
            image = self.image_unpatch_emb(image)
            x = image.unsqueeze(2).permute(0, 2, 1, 3, 4)
        return x


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
            self.q_step_counter += x.size(0)
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, x.size(1))
            self.reservoir = x_flat if self.reservoir is None else torch.cat([self.reservoir, x_flat], dim=0)
            self.reservoir = self.reservoir[torch.randperm(self.reservoir.size(0))[:self.reservoir_size]].detach()
            if self.q_step_counter < self.q_init:
                qe, commit_loss, indices = x, x.new_tensor(0), None
            else:
                # if self.q_step_counter < self.q_init + self.q_refresh_end:
                #     if (
                #             self.q_step_counter + self.q_init) % self.q_refresh_step == 0 or self.q_step_counter == self.q_init or self.q_step_counter == self.q_init + self.q_refresh_end - 1:
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


class VIVIT(nn.Module):
    def __init__(self, patch_size=(5, 8, 8), compressed_frames=20, latent_size=32, c_hidden=64, c_codebook=16,
                 codebook_size=1024, num_layers_enc=4, num_layers_dec=4, num_heads=4):
        super().__init__()
        self.encoder = Encoder(patch_size=patch_size, hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames,
                               num_layers=num_layers_enc, num_heads=num_heads)
        self.cod_mapper = nn.Linear(c_hidden, c_codebook)
        self.batchnorm = nn.BatchNorm2d(c_codebook)

        self.cod_unmapper = nn.Linear(c_codebook, c_hidden)
        self.decoder = Decoder(patch_size=patch_size, hidden_channels=c_hidden, size=latent_size, compressed_frames=compressed_frames,
                               num_layers=num_layers_dec, num_heads=num_heads)

        self.codebook_size = codebook_size
        self.vqmodule = VQModule(
            c_codebook, k=codebook_size,
            q_init=0, q_refresh_step=15010, q_refresh_end=15010 * 130
            # q_init=15010 * 20, q_refresh_step=15010, q_refresh_end=15010 * 130
        )

    def encode(self, image, video):
        x = self.encoder(image, video)  # B x T x (H x W) x C
        x = self.cod_mapper(x)
        x = self.batchnorm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        qe, commit_loss, indices = self.vqmodule(x, dim=-1)
        return (x, qe), commit_loss, indices

    def decode(self, x):
        x = self.cod_unmapper(x)
        x = self.decoder(x)
        return x

    def decode_indices(self, x):
        return self.decode(self.vqmodule.vquantizer.idx2vq(x, dim=-1))

    def forward(self, image, video=None):
        (_, qe), commit_loss, _ = self.encode(image, video)
        decoded = self.decode(qe)
        return decoded, commit_loss


if __name__ == '__main__':
    device = "cpu"
    vq = VIVIT(latent_size=16, compressed_frames=5, patch_size=(2, 8, 8)).to(device)
    print(sum([p.numel() for p in vq.parameters()]))
    image = torch.randn(1, 3, 128, 128).to(device)
    video = torch.randn(1, 10, 3, 128, 128).to(device)
    r = vq(image, video)[0]
    # r = vq(image)[0]
    print(r.shape)