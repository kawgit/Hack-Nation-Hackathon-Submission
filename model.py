import torch
import torch.nn as nn
import torch.nn.functional as F
class BrainWaveIntentModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.eeg_dim = 6
        self.moment_dim = 720
        self.num_classes = 5

        self.eeg_

    def forward(self, eeg_features, moment_features):
        pass

class MultiResEmbedding(nn.Module):
    def __init__(self, n_channels, resolutions, dim):
        super().__init__()
        self.n_channels = n_channels
        self.resolutions = resolutions
        self.dim = dim
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_channels * (res + 1), dim) for res in resolutions
        ])
        
        for i, res in enumerate(resolutions):
            self.register_buffer(f"b_{i}", torch.linspace(0, 1, res))
            self.register_buffer(f"o_{i}", torch.arange(n_channels) * (res + 1))

    def forward(self, x):
        out = []
        for i in range(len(self.resolutions)):
            idx = torch.bucketize(x, getattr(self, f"b_{i}")) + getattr(self, f"o_{i}")
            out.append(self.embeddings[i](idx))
        return torch.cat(out, dim=-1)
