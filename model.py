import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResEmbedding(nn.Module):
    def __init__(self, n_channels, resolutions, dim):
        super().__init__()
        self.n_channels = n_channels
        self.resolutions = resolutions
        self.bags = nn.ModuleList([
            nn.EmbeddingBag(n_channels * (res + 1), dim, mode='sum') 
            for res in resolutions
        ])
        for i, res in enumerate(resolutions):
            self.register_buffer(f"b_{i}", torch.linspace(0, 1, res))
            self.register_buffer(f"o_{i}", torch.arange(n_channels) * (res + 1))

    def forward(self, x):
        out = 0
        for i in range(len(self.resolutions)):
            idx = torch.bucketize(x, getattr(self, f"b_{i}")) + getattr(self, f"o_{i}")
            out += self.bags[i](idx)
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class SDPA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def rope(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos(), emb.sin()
        x2 = torch.stack([-x[..., 1::2], x[..., 0::2]], dim=-1).flatten(-2)
        return (x * cos) + (x2 * sin)

    def forward(self, x):
        q = self.rope(x)
        k = self.rope(x)
        return F.scaled_dot_product_attention(q, k, x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SDPA(dim)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BrainWaveIntentModel(nn.Module):
    def __init__(self, num_layers=4, dim=32, mlp_ratio=4):
        super().__init__()
        self.eeg_table = MultiResEmbedding(6, [4, 8, 16], dim)
        self.moment_table = MultiResEmbedding(720, [4, 8, 16], dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, dim * mlp_ratio) for _ in range(num_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, 5)

    def forward(self, eeg, moment):
        eeg_x = self.eeg_table(eeg).unsqueeze(1)
        mom_x = self.moment_table(moment).unsqueeze(1)
        x = torch.cat([eeg_x, mom_x], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.final_norm(x).mean(1))