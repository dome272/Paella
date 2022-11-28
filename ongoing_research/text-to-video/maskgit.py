"""
Code from lucidrains https://github.com/lucidrains/phenaki-pytorch/blob/main/phenaki_pytorch/phenaki_pytorch.py and modified
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def log(t, eps=1e-20):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


def top_k(logits, thres=0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs


def FeedForward(dim, mult=4):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_context=None,
            dim_head=64,
            heads=8,
            causal=False,
            num_null_kv=2
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
            self,
            x,
            mask=None,
            context=None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        q = q * self.scale

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]

        if exists(mask):
            mask = F.pad(mask, (j - mask.shape[-1], 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
            self,
            dim,
            *,
            depth,
            dim_context=None,
            causal=False,
            dim_head=64,
            heads=8,
            ff_mult=4,
            attn_num_null_kv=2,
            has_cross_attn=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, causal=causal),
                Attention(dim=dim, dim_head=dim_head, dim_context=dim_context, heads=heads, causal=False,
                          num_null_kv=attn_num_null_kv) if has_cross_attn else None,
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context=None, mask=None):

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=None) + x

            x = ff(x) + x

        return self.norm_out(x)


class MaskGit(nn.Module):
    def __init__(
            self,
            *,
            dim,
            num_tokens,
            max_seq_len,
            **kwargs
    ):
        super().__init__()
        self.mask_id = num_tokens

        self.token_emb = nn.Embedding(num_tokens + 1, dim)  # last token is used as mask_id
        self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.transformer = Transformer(
            dim=dim,
            attn_num_null_kv=2,
            has_cross_attn=True,
            **kwargs
        )

        self.to_logits = nn.Linear(dim, num_tokens)

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def gamma(self, r):
        return (r * torch.pi / 2).cos()

    def add_noise(self, x, r):
        r = self.gamma(r)[:, None]
        mask = torch.bernoulli(r * torch.ones_like(x))
        mask = mask.round().bool()
        x = x * (~mask) + self.mask_id * mask
        return x, mask

    def forward(self, x, text_cond=None, **kwargs):
        b, n = x.shape

        x = self.token_emb(x)
        x = self.pos_emb(torch.arange(n, device=x.device)) + x

        x = self.transformer(x, text_cond, **kwargs)

        return self.to_logits(x)

    def loss(self, pred, video_indices, mask):
        acc = (pred.permute(0, 2, 1).argmax(1) == video_indices).float().mean()
        video_indices = video_indices[mask]  # 839
        # video_indices = video_indices.flatten()
        mask = mask.flatten()  # 1536
        pred = pred.view(-1, pred.shape[-1])[mask]  # 839x1024
        # pred = pred.view(-1, pred.shape[-1])  # 839x1024
        return self.loss_fn(pred, video_indices), acc


class MaskGitTrainWrapper(nn.Module):
    def __init__(
            self,
            maskgit,
            *,
            steps
    ):
        super().__init__()
        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id

        self.steps = steps

    def forward(self, x, **kwargs):
        batch, seq, device = *x.shape, x.device

        self.maskgit.train()

        rand_step = torch.randint(0, self.steps, (1,), device=device)
        num_tokens_mask = (seq * torch.cos(rand_step * math.pi * 0.5 / self.steps)).round().long().clamp(
            min=1)  # cosine schedule was best

        _, indices = torch.randn((batch, seq), device=device).topk(num_tokens_mask.item(), dim=-1)
        mask = torch.zeros((batch, seq), device=device).scatter(1, indices, 1.).bool()

        masked_input = torch.where(mask, self.mask_id, x)

        logits = self.maskgit(masked_input, **kwargs)

        loss = F.cross_entropy(logits[mask], x[mask])
        return loss


class Phenaki(nn.Module):
    def __init__(
            self,
            *,
            maskgit: MaskGit,
            steps=18,  # 18 is the ideal steps with token critic
            sample_temperature=0.,
            text_embed_dim=None,
            cond_drop_prob=0.25,
            max_text_len=128
    ):
        super().__init__()

        self.maskgit = maskgit
        self.mask_id = maskgit.mask_id
        self.maskgit_trainer = MaskGitTrainWrapper(maskgit, steps=steps)

        # sampling

        self.steps = steps
        self.sample_temperature = sample_temperature

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob  # classifier free guidance for transformers - @crowsonkb

    @torch.no_grad()
    def sample( self, *, text, num_frames, cond_scale=3., starting_temperature=0.9, noise_K=1.):
        device = next(self.parameters()).device
        num_tokens = self.cvivit.num_tokens_per_frames(num_frames)

        with torch.no_grad():
            text_embeds = self.encode_texts([text], output_device=device)
            text_mask = torch.any(text_embeds != 0, dim=-1)

        shape = (1, num_tokens)

        video_token_ids = torch.full(shape, self.mask_id, device=device)
        mask = torch.ones(shape, device=device, dtype=torch.bool)

        scores = None

        for step in range(self.steps):
            is_first_step = step == 0
            is_last_step = step == (self.steps - 1)

            steps_til_x0 = self.steps - (step + 1)

            if not is_first_step and exists(scores):
                time = torch.full((1,), step / self.steps, device=device)
                num_tokens_mask = (num_tokens * torch.cos(time * math.pi * 0.5)).round().long().clamp(min=1)

                _, indices = scores.topk(num_tokens_mask.item(), dim=-1)
                mask = torch.zeros(shape, device=device).scatter(1, indices, 1).bool()

            video_token_ids = torch.where(mask, self.mask_id, video_token_ids)

            logits = self.maskgit.forward_with_cond_scale(
                video_token_ids,
                context=text_embeds,
                mask=text_mask,
                cond_scale=cond_scale
            )

            temperature = starting_temperature * (step / steps_til_x0)
            pred_video_ids = gumbel_sample(logits, temperature=temperature)

            video_token_ids = torch.where(mask, pred_video_ids, video_token_ids)

            if not is_last_step:
                scores = logits.gather(2, rearrange(pred_video_ids, '... -> ... 1'))
                scores = 1 - rearrange(scores, '... 1 -> ...')
                scores = torch.where(mask, scores, -1e4)

        video = self.cvivit.decode_from_codebook_indices(video_token_ids)
        return video


if __name__ == '__main__':
    # m = MaskGit(dim=2048, num_tokens=1024, max_seq_len=1536, depth=32, dim_context=512, heads=32)  # paper version
    m = MaskGit(dim=1224, num_tokens=8192, max_seq_len=1536, depth=22, dim_context=512, heads=22)  # 6 x 16 x 16
    # print(sum([p.numel() for p in m.parameters()]))
    for name, p in m.named_parameters():
        print(name)
        print(p.shape)
    # x = torch.randint(low=0, high=8192, size=(6, 16, 16)).view(1, -1)
    # text_cond = torch.randn(1, 10, 512)
    # print(m(x, text_cond).shape)