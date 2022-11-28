import math
import numpy as np
import torch
import torch.nn as nn


class EMA:
    def __init__(self, beta, step=0):
        super().__init__()
        self.beta = beta
        self.step = step

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def geglu(x):
    assert x.size(-1) % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * nn.functional.gelu(b)


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


class ResBlock(nn.Module):
    def __init__(self, c, c_hidden, c_cond=0, c_skip=0, scaler=None, layer_scale_init_value=1e-6):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(c, c, kernel_size=3, groups=c)
        )
        self.ln = ModulatedLayerNorm(c, channels_first=False)
        self.channelwise = nn.Sequential(
            nn.Linear(c+c_skip, c_hidden),
            nn.GELU(),
            nn.Linear(c_hidden, c),
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c),  requires_grad=True) if layer_scale_init_value > 0 else None
        self.scaler = scaler
        if c_cond > 0:
            self.cond_mapper = nn.Linear(c_cond, c)

    def forward(self, x, s=None, skip=None):
        res = x
        x = self.depthwise(x)
        if s is not None:
            if s.size(2) == s.size(3) == 1:
                s = s.expand(-1, -1, x.size(2), x.size(3))
            s = self.cond_mapper(s.permute(0, 2, 3, 1))
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


class DenoiseGIC(nn.Module):
    def __init__(self, num_labels, c_hidden=1024, c_r=64, layers=32):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.embedding = nn.Embedding(num_labels, c_hidden)
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            block = ResBlock(c_hidden, c_hidden*4, 768+c_r)
            block.channelwise[-1].weight.data *= np.sqrt(1 / layers)
            self.blocks.append(block)
        self.clf = nn.Conv2d(c_hidden, num_labels, kernel_size=1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r, random_x=None):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(r * torch.ones_like(x), )
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.randint_like(x, 0, self.num_labels)
        x = x * (1-mask) + random_x * mask
        return x, mask

    def gen_r_embedding(self, r, max_positions=10000):
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x, c, r): # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        if self.training:
            x = self.embedding(x).permute(0, 3, 1, 2).contiguous()
        else:
             x = self.embedding(x).permute(0, 3, 1, 2)
        for block in self.blocks:
            s = torch.cat([c, r_embed], dim=1)[:, :, None, None]
            x = block(x, s)
        x = self.clf(x)
        return x


def normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class DenoiseUNet(nn.Module):
    def __init__(self, num_labels, c_hidden=1720, c_clip=1024, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4]):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels, c_levels[0])

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for j in range(num_blocks):
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
                    nn.ConvTranspose2d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=4,
                                       stride=2, padding=1))
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
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
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
        s = torch.cat([c, r_embed], dim=-1)[:, :, None, None]
        level_outputs = self._down_encode_(x, s)
        x = self._up_decode(level_outputs, s)
        x = self.clf(x)
        return x


class DenoiseUNetT5(nn.Module):
    def __init__(self, num_labels, c_hidden=1280, c_t5=512, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4],
                 attpool_heads=8, attnpool_layers=1):
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
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_t5 + c_r)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 1 - i] * 4, 512 + c_r,
                                 c_levels[len(c_levels) - 1 - i] if (j == 0 and i > 0) else 0)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=4,
                                       stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_labels, kernel_size=1)

        # ATTN POOLING
        self.pool_queries = nn.Parameter(torch.randn(1, sum(down_levels) + sum(up_levels), c_t5))
        attn_layer = nn.TransformerEncoderLayer(c_t5, attpool_heads, dim_feedforward=c_t5 * 4, batch_first=True,
                                                norm_first=True, activation='gelu')
        for layer in attn_layer.children():
            if isinstance(layer, nn.Linear):
                layer.bias = None
        self.attnpool = nn.TransformerEncoder(attn_layer, num_layers=attnpool_layers, norm=nn.LayerNorm(c_t5))

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
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    s_level = s[:, 0]
                    s = s[:, 1:]
                    x = block(x, s_level)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    s_level = s[:, 0]
                    s = s[:, 1:]
                    if i > 0 and j == 0:
                        x = block(x, s_level, level_outputs[i])
                    else:
                        x = block(x, s_level)
                else:
                    x = block(x)
        return x

    def _attn_pool(self, x):
        x = torch.cat([x, self.pool_queries.expand(x.size(0), -1, -1)], dim=1)
        x = self.attnpool(x)[:, -self.pool_queries.size(1):]
        return x

    def forward(self, x, c, r):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        c = self._attn_pool(c)
        s = torch.cat([c, r_embed.unsqueeze(1).expand(-1, c.size(1), -1)], dim=-1)[:, :, :, None, None]
        level_outputs = self._down_encode_(x, s[:, :sum(self.down_levels)])
        x = self._up_decode(level_outputs, s[:, sum(self.down_levels):])
        x = self.clf(x)
        return x


class MaskedUNet(nn.Module):
    def __init__(self, num_labels, c_hidden=1280, c_t5=512, c_r=64, down_levels=[4, 8, 16], up_levels=[16, 8, 4],
                 attpool_heads=8, attnpool_layers=1):
        super().__init__()
        self.num_labels = num_labels
        self.c_r = c_r
        self.down_levels = down_levels
        self.up_levels = up_levels
        c_levels = [c_hidden // (2 ** i) for i in reversed(range(len(down_levels)))]
        self.embedding = nn.Embedding(num_labels + 1, c_levels[0])  # <--- MaskGIC change
        self.mask_idx = num_labels  # <--- MaskGIC change

        # DOWN BLOCKS
        self.down_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(down_levels):
            blocks = []
            if i > 0:
                blocks.append(nn.Conv2d(c_levels[i - 1], c_levels[i], kernel_size=4, stride=2, padding=1))
            for _ in range(num_blocks):
                block = ResBlock(c_levels[i], c_levels[i] * 4, c_t5 + c_r)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(down_levels))
                blocks.append(block)
            self.down_blocks.append(nn.ModuleList(blocks))

        # UP BLOCKS
        self.up_blocks = nn.ModuleList()
        for i, num_blocks in enumerate(up_levels):
            blocks = []
            for j in range(num_blocks):
                block = ResBlock(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 1 - i] * 4, 512 + c_r,
                                 c_levels[len(c_levels) - 1 - i] if (j == 0 and i > 0) else 0)
                block.channelwise[-1].weight.data *= np.sqrt(1 / sum(up_levels))
                blocks.append(block)
            if i < len(up_levels) - 1:
                blocks.append(
                    nn.ConvTranspose2d(c_levels[len(c_levels) - 1 - i], c_levels[len(c_levels) - 2 - i], kernel_size=4,
                                       stride=2, padding=1))
            self.up_blocks.append(nn.ModuleList(blocks))

        self.clf = nn.Conv2d(c_levels[0], num_labels, kernel_size=1)

        # ATTN POOLING
        self.pool_queries = nn.Parameter(torch.randn(1, sum(down_levels) + sum(up_levels), c_t5))
        attn_layer = nn.TransformerEncoderLayer(c_t5, attpool_heads, dim_feedforward=c_t5 * 4, batch_first=True,
                                                norm_first=True, activation='gelu')
        for layer in attn_layer.children():
            if isinstance(layer, nn.Linear):
                layer.bias = None
        self.attnpool = nn.TransformerEncoder(attn_layer, num_layers=attnpool_layers, norm=nn.LayerNorm(c_t5))

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r, random_x=None):
        r = self.gamma(r)[:, None, None]
        mask = torch.bernoulli(r * torch.ones_like(x), )
        mask = mask.round().long()
        if random_x is None:
            random_x = torch.full_like(x, self.mask_idx)  # <--- MaskGIC change
        x = x * (1 - mask) + random_x * mask
        return x, mask

    def gen_r_embedding(self, r, max_positions=10000):
        r = self.gamma(r) * max_positions
        half_dim = self.c_r // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=r.device).float().mul(-emb).exp()
        emb = r[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.c_r % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def _down_encode_(self, x, s):
        level_outputs = []
        for i, blocks in enumerate(self.down_blocks):
            for block in blocks:
                if isinstance(block, ResBlock):
                    s_level = s[:, 0]
                    s = s[:, 1:]
                    x = block(x, s_level)
                else:
                    x = block(x)
            level_outputs.insert(0, x)
        return level_outputs

    def _up_decode(self, level_outputs, s):
        x = level_outputs[0]
        for i, blocks in enumerate(self.up_blocks):
            for j, block in enumerate(blocks):
                if isinstance(block, ResBlock):
                    s_level = s[:, 0]
                    s = s[:, 1:]
                    if i > 0 and j == 0:
                        x = block(x, s_level, level_outputs[i])
                    else:
                        x = block(x, s_level)
                else:
                    x = block(x)
        return x

    def _attn_pool(self, x):
        x = torch.cat([x, self.pool_queries.expand(x.size(0), -1, -1)], dim=1)
        x = self.attnpool(x)[:, -self.pool_queries.size(1):]
        return x

    def forward(self, x, c, r):  # r is a uniform value between 0 and 1
        r_embed = self.gen_r_embedding(r)
        x = self.embedding(x).permute(0, 3, 1, 2)
        c = self._attn_pool(c)
        s = torch.cat([c, r_embed.unsqueeze(1).expand(-1, c.size(1), -1)], dim=-1)[:, :, :, None, None]
        level_outputs = self._down_encode_(x, s[:, :sum(self.down_levels)])
        x = self._up_decode(level_outputs, s[:, sum(self.down_levels):])
        x = self.clf(x)
        return x

if __name__ == '__main__':
    device = "cuda"
    # from transformers import T5Tokenizer, T5Model
    # captions = ["hey this is funny", "bruh what"]
    # t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")  # change with "t5-b3" for the 10GB model LoL
    # t5_model = T5Model.from_pretrained("t5-small").to(device).requires_grad_(False)
    # text_tokens = t5_tokenizer(captions, return_tensors="pt", padding=True, truncation=True).input_ids
    # text_tokens = text_tokens.to(device)
    # text_embeddings = t5_model.encoder(input_ids=text_tokens).last_hidden_state
    # print(text_embeddings.shape)
    # exit()

    # model = DenoiseUNetT5(1024)
    # x = torch.randint(0, 1024, (1, 32, 32)).long()
    # c = torch.randn((1, 6, 512))
    # r = torch.rand(1)
    # model(x, c, r)

    model = DenoiseUNet(8192, c_hidden=1280, down_levels=[1, 2, 8, 32], up_levels=[32, 8, 2, 1]).to(device)
    # model = DenoiseUNet(8192).to(device)
    print(sum([p.numel() for p in model.parameters()]))
    x = torch.randint(0, 1024, (1, 32, 32)).long().to(device)
    c = torch.randn((1, 1024)).to(device)
    r = torch.rand(1).to(device)
    torch.cuda.synchronize()
    import time
    t = time.time()
    for i in range(100):
        model(x, c, r)
    print(time.time()-t)

