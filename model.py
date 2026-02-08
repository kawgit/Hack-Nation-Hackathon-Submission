import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiResEmbedding(nn.Module):
    def __init__(self, n_channels, resolutions, dim):
        super().__init__()
        self.n_channels = n_channels
        self.resolutions = torch.tensor(resolutions)
        
        res_sizes = n_channels * (self.resolutions + 1)
        total_vocab_size = res_sizes.sum().item()
        
        self.bag = nn.EmbeddingBag(total_vocab_size, dim, mode='sum')
        
        global_offsets = torch.cat([torch.tensor([0]), res_sizes.cumsum(0)[:-1]])
        self.register_buffer("global_offsets", global_offsets)
        
        for i, res in enumerate(resolutions):
            self.register_buffer(f"b_{i}", torch.linspace(0, 1, res))
            self.register_buffer(f"o_{i}", torch.arange(n_channels) * (res + 1))

    def forward(self, features):
        indices = []
        for i in range(len(self.resolutions)):
            idx = torch.bucketize(features, getattr(self, f"b_{i}")) + getattr(self, f"o_{i}")
            indices.append(idx + self.global_offsets[i])
        indices = torch.cat(indices, dim=-1).reshape(features.size(0), -1)
        return self.bag(indices)

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

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x):
        t = torch.arange(x.shape[1], device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (x * emb.cos()) + (self.rotate_half(x) * emb.sin())

    def forward(self, x):
        q, k = self.apply_rope(x), self.apply_rope(x)
        return F.scaled_dot_product_attention(q, k, x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim):
        super().__init__()
        self.norm1, self.attn = RMSNorm(dim), SDPA(dim)
        self.norm2, self.mlp = RMSNorm(dim), MLP(dim, mlp_dim)

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
        self.final_norm, self.head = RMSNorm(dim), nn.Linear(dim, 5)

    def forward(self, eeg_features, moment_features):
        eeg_features = self.eeg_table(eeg_features)
        moment_features = self.moment_table(moment_features)
        features = torch.stack([eeg_features, moment_features], dim=1)
        for layer in self.layers:
            features = layer(features)
        return self.head(self.final_norm(features).mean(1))