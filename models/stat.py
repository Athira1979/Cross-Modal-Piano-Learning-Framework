 
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class PATA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # scoring function f(y_t)
        self.score = nn.Linear(dim, 1)

        # Multi-head attention (as described)
        self.mha = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        # Temporal Contextual FFN (TCFFN)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (B, T, D)

        # Multi-head self-attention
        attn_out, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_out)

        # Scoring function → e_t
        e = self.score(x).squeeze(-1)  # (B, T)

        # Softmax → α_t
        alpha = F.softmax(e, dim=1).unsqueeze(-1)  # (B, T, 1)

        # Weighted sum → ŷ
        y_hat = torch.sum(alpha * x, dim=1)  # (B, D)

        # TCFFN
        out = self.ffn(y_hat)
        out = self.norm2(out)

        return out, alpha

class PASA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # scoring function g(x_j)
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x shape: (B, J, D)

        # Swish activation (non-linearized spectral encoding)
        x = x * torch.sigmoid(x)

        # e_j scores
        e = self.score(x).squeeze(-1)  # (B, J)

        # γ_j (softmax over joints)
        gamma = F.softmax(e, dim=1).unsqueeze(-1)  # (B, J, 1)

        # weighted sum
        X_hat = torch.sum(gamma * x, dim=1)  # (B, D)

        return X_hat, gamma

class STATBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.pata = PATA(dim)
        self.pasa = PASA(dim)

    def forward(self, temporal_input, spatial_input):
        # temporal_input: (B, T, D)
        # spatial_input: (B, J, D)

        temporal_out, alpha = self.pata(temporal_input)
        spatial_out, gamma = self.pasa(spatial_input)

        # Combine both attentions
        combined = temporal_out + spatial_out

        return combined, alpha, gamma

