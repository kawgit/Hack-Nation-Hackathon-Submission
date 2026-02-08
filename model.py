import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiResEmbedding(nn.Module):
    """
    Stable embedding for categorical or bucketable features (e.g., EEG channels).
    """
    def __init__(self, n_channels, resolutions, dim):
        super().__init__()
        self.n_channels = n_channels
        if isinstance(resolutions, int): resolutions = [resolutions]
        self.resolutions = torch.tensor(resolutions)
        
        # Calculate total vocab size based on bins per channel
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

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class SDPA(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1, max_len=1024):
        super().__init__()
        assert dim % num_heads == 0, "Dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

        # --- RoPE Initialization ---
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        self.max_len_cached = max_len
        t = torch.arange(self.max_len_cached).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def rotate_half(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope(self, x):
        seq_len = x.shape[1]
        if seq_len > self.max_len_cached:
            self.max_len_cached = seq_len
            t = torch.arange(self.max_len_cached, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]

        cos = self.cos_cached[:, :, :seq_len, ...].transpose(1, 2)
        sin = self.sin_cached[:, :, :seq_len, ...].transpose(1, 2)

        x_float = x.float() 
        x_rotated = (x_float * cos) + (self.rotate_half(x_float) * sin)
        return x_rotated.type_as(x)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q = self.apply_rope(q)
        k = self.apply_rope(k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.dim)
        return self.o_proj(output)

class TransformerBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = SDPA(dim, dropout=dropout)
        self.norm2 = nn.RMSNorm(dim)
        self.mlp = MLP(dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class BrainWaveIntentModel(nn.Module):
    def __init__(self, num_layers=4, dim=64, mlp_ratio=4, num_classes=5):
        super().__init__()
        
        # 1. EEG Encoder
        self.eeg_table = MultiResEmbedding(n_channels=6, resolutions=[8], dim=dim)
        
        # 2. Moment Encoder (Linear Projection)
        # Accepts (Batch, Seq_Len, 720) OR (Batch, 720)
        self.moment_encoder = nn.Sequential(
            nn.LayerNorm(720),
            nn.Linear(720, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.1)
        )
        
        # 3. Learnable Type Embeddings
        self.type_embed = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        
        # 4. Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(dim, dim * mlp_ratio) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.EmbeddingBag):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, eeg_features, moment_features):
        # 1. Embed EEG -> (Batch, Dim)
        eeg_emb = self.eeg_table(eeg_features)
        
        # 2. Embed Moments
        # Input: (Batch, 72, 720)
        # Output: (Batch, 72, Dim)
        moment_emb = self.moment_encoder(moment_features) 
        
        # 3. Pooling: Compress time sequence into one vector
        # Output: (Batch, Dim)
        if moment_emb.dim() == 3:
            moment_emb = moment_emb.mean(dim=1)
        
        # 4. Create Sequence: (Batch, 2, Dim)
        x = torch.stack([eeg_emb, moment_emb], dim=1)
        
        # 5. Add Type Embedding
        x = x + self.type_embed
        
        # 6. Transformer Layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        
        # 7. Classification
        return self.head(x.mean(dim=1))