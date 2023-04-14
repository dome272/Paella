import torch
from torch import nn

class CustomMultiheadAttention(nn.MultiheadAttention):
    def forward(self, *args, attn_weights=None, **kwargs):
        q, k, v = args[:3]
        need_weights = kwargs.get('need_weights', False)
        
        w = self.in_proj_weight.chunk(3, dim=0)
        b = self.in_proj_bias.chunk(3, dim=0)
        
        if not self.batch_first:
            q, k, v = q.permute(0, 1), k.permute(0, 1), v.permute(0, 1)
        
        q = nn.functional.linear(q, w[0], bias=b[0]).view(q.size(0), q.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        k = nn.functional.linear(k, w[1], bias=b[1]).view(k.size(0), k.size(1), self.num_heads, -1).permute(0, 2, 1, 3)
        v = nn.functional.linear(v, w[2], bias=b[2]).view(v.size(0), v.size(1), self.num_heads, -1).permute(0, 2, 1, 3)

        scores = (q @ k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        attention = scores.softmax(dim=-1)
        # print(attention.shape)
        
        if attn_weights is not None:
            # print("q ", q.shape)
            # print("k ", k.shape)
            weights = torch.ones((attention.shape[2], attention.shape[3])).to(q.device)
            # print("Weights: ", weights.shape)
            attn_weights = attn_weights.expand(attention.shape[2], attn_weights.shape[0])
            weights[-attn_weights.shape[0]:, -attn_weights.shape[1]:] = attn_weights
            # print(f"{-attn_weights.shape[0]}, {-attn_weights.shape[1]}")
            attn_weights = weights.clone()
            # print("Attn Weights: ", weights.shape)
            # print("weight", attn_weights.shape)
            attention = attention * attn_weights
        
        x = attention @ v
        x = x.permute(0, 2, 1, 3).reshape(x.size(0), x.size(2), -1)
        x = self.out_proj(x)
        
        if not self.batch_first:
            x = x.permute(0, 1)
        
        return (x, attention if need_weights else None) 
    
def replace_attention_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_attention_layers(module)
            
        if isinstance(module, nn.MultiheadAttention):
            new_module = CustomMultiheadAttention(module.embed_dim, module.num_heads, dropout=module.dropout, bias=True, batch_first=module.batch_first)
            new_module.load_state_dict(module.state_dict())
            setattr(model, n, new_module)