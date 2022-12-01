import math
import numpy as np
import torch
import torch.nn as nn


class ModulatedLayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, channels_first=True):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps)
        self.gamma = nn.Parameter(torch.randn(1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, 1, 1))
        self.channels_first = channels_first

    def forward(self, x, w=None):
        x = x.permute(0, 2, 3, 1) if self.channels_first else x
        if w is None:
            x = self.ln(x)
        else:
            x = self.gamma * w * self.ln(x) + self.beta * w
        x = x.permute(0, 3, 1, 2) if self.channels_first else x
        return x

class Attention2D(nn.Module):
    def __init__(self, c, nhead=8):
        super().__init__()
        self.ln = nn.LayerNorm(c)
        self.attn = torch.nn.MultiheadAttention(c, nhead, bias=True, batch_first=True)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1) # Bx4xHxW -> Bx(HxW)x4
        norm_x = self.ln(x) 
        x = x + self.attn(norm_x, norm_x, norm_x, need_weights=False)[0]
        x = x.permute(0, 2, 1).view(*orig_shape)
        return x

class ResBlock(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-6, use_attention=False):
        super().__init__()
        if use_attention:
            self.depthwise = Attention2D(c)
        else:
            self.depthwise = nn.Sequential(
                nn.ReflectionPad2d(1),
                nn.Conv2d(c, c, kernel_size=3, groups=c)
            )
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        self.channelwise = nn.Sequential(
            nn.Linear(c + c_skip, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c), requires_grad=True) if layer_scale_init_value > 0 else None
        self.scaler = scaler
        if c_cond > 0:
            self.cond_mapper = nn.Linear(c_cond, c)

    def forward(self, x, s=None, skip=None):
        res = x
        x = self.depthwise(x)
        if s is not None:
            if s.size(2) == s.size(3) == 1:
                s = s.expand(-1, -1, x.size(2), x.size(3))
            elif s.size(2) != x.size(2) or s.size(3) != x.size(3):
                s = nn.functional.interpolate(s, size=x.shape[-2:], mode='bilinear')
            s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # s = self.cond_mapper(s.permute(0, 2, 3, 1))
            # if s.size(1) == s.size(2) == 1:
            #     s = s.expand(-1, x.size(2), x.size(3), -1)
        x = self.ln(x.permute(0, 2, 3, 1), s)
        if skip is not None:
            x = torch.cat([x, skip.permute(0, 2, 3, 1)], dim=-1)
        x = self.channelwise(x)
        x = self.gamma * x if self.gamma is not None else x
        x = res + x.permute(0, 3, 1, 2)
        if self.scaler is not None:
            x = self.scaler(x)
        return x


class DenoiseUNet(nn.Module):
    def __init__(self, num_labels, c_hidden=1280, c_clip=1024, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4]):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels, c_levels[0])

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for _ in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_clip + c_r)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 1 - i] * 4, c_clip + c_r,
                                 c_levels[len(c_levels) - 1 - i] if (j == 0 and i > 0) else 0)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_labels, kernel_size=1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r, random_x=None):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(r * torch.ones_like(x), )
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def gen_r_embedding(self, r, max_positions=10000):
        dtype = r.dtype
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb.to(dtype)

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    x = block(x, s)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
        return x

    def forward(self, x, c, r):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        if len(c.shape) == 2:
            s = torch.cat([c, r_embed], dim=-1)[:, :, None, None]
        else:
            r_embed = r_embed[:, :, None, None].expand(-1, -1, c.size(2), c.size(3))
            s = torch.cat([c, r_embed], dim=1)
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x

class DenoiseUNet_selfattn(nn.Module):
    def __init__(self, num_labels, c_hidden=1280, c_clip=1024, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4], down_attn=[[], [6], [10, 12, 14]], up_attn=[[1, 3, 5], [1], []]):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels, c_levels[0])

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for j in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_clip + c_r, use_attention=j in down_attn[i])
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 1 - i] * 4, c_clip + c_r,
                                 c_levels[len(c_levels) - 1 - i] if (j == 0 and i > 0) else 0, use_attention=j in up_attn[i])
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_labels, kernel_size=1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r, random_x=None):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(r * torch.ones_like(x), )
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def gen_r_embedding(self, r, max_positions=10000):
        dtype = r.dtype
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb.to(dtype)

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    x = block(x, s)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    # s_level = s[:, 0]
                    # s = s[:, 1:]
                    if i > 0 and j == 0:
                        x = block(x, s, level_outputs[i])
                    else:
                        x = block(x, s)
                else:
                    x = block(x)
        return x

    def forward(self, x, c, r):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        if len(c.shape) == 2:
            s = torch.cat([c, r_embed], dim=-1)[:, :, None, None]
        else:
            r_embed = r_embed[:, :, None, None].expand(-1, -1, c.size(2), c.size(3))
            s = torch.cat([c, r_embed], dim=1)
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x

if __name__ == '__main__':
    device = "cuda"
    model = DenoiseUNet(1024).to(device)
    print(sum([p.numel() for p in model.parameters()]))
    x = torch.randint(0, 1024, (1, 32, 32)).long().to(device)
    c = torch.randn((1, 1024)).to(device)
    r = torch.rand(1).to(device)
    model(x, c, r)

