 
import torch
import torch.nn as nn


class SharedEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, Y):
        # Z = WY + b
        return self.proj(Y)


class MHGCA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, target, source):
        # Cross-attention
        attn_out, _ = self.attn(target, source, source)

        # Gating (important for your paper)
        gate = self.gate(attn_out)
        gated = attn_out * gate

        # Projection + LayerNorm
        out = self.proj(gated)
        out = self.norm(out + target)

        return out

class InterModalStage(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Cross interactions (matches equations 10–17)
        self.a_from_g = MHGCA(dim)
        self.a_from_p = MHGCA(dim)

        self.g_from_a = MHGCA(dim)
        self.g_from_p = MHGCA(dim)

        self.p_from_a = MHGCA(dim)
        self.p_from_g = MHGCA(dim)

    def forward(self, Ya, Yg, Yp):
        # Audio attends to gesture + posture
        Ya = self.a_from_g(Ya, Yg)
        Ya = self.a_from_p(Ya, Yp)

        # Gesture attends to others
        Yg = self.g_from_a(Yg, Ya)
        Yg = self.g_from_p(Yg, Yp)

        # Posture attends to others
        Yp = self.p_from_a(Yp, Ya)
        Yp = self.p_from_g(Yp, Yg)

        return Ya, Yg, Yp

class CMGSA(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(dim, 4, batch_first=True)

        self.gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # CrossRefine FFN (FFN_CR)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.self_attn(x, x, x)

        # Gating
        g = self.gate(attn_out)
        x = x + attn_out * g
        x = self.norm1(x)

        # FFN_CR
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class IntraModalStage(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.audio = CMGSA(dim)
        self.gesture = CMGSA(dim)
        self.posture = CMGSA(dim)

    def forward(self, Ya, Yg, Yp):
        Ya = self.audio(Ya)
        Yg = self.gesture(Yg)
        Yp = self.posture(Yp)

        return Ya, Yg, Yp

class FusionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.weights = nn.Parameter(torch.ones(3))  # dynamic weighting

        self.proj = nn.Linear(dim, dim)

    def forward(self, Ya, Yg, Yp):
        w = torch.softmax(self.weights, dim=0)

        fused = (
            w[0] * Ya +
            w[1] * Yg +
            w[2] * Yp
        )

        return self.proj(fused)


import torch.nn.functional as F

def align_time(x, target_len):
    """
    x: (B, T, D)
    return: (B, target_len, D)
    """
    x = x.permute(0, 2, 1)              # (B, D, T)
    x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
    x = x.permute(0, 2, 1)              # (B, T, D)
    return x

class CMTPF(nn.Module):
    def __init__(self, da, dg, dp, d_model=128):
        super().__init__()

        # Shared embeddings (Eq. 9)
        self.embed_a = SharedEmbedding(da, d_model)
        self.embed_g = SharedEmbedding(dg, d_model)
        self.embed_p = SharedEmbedding(dp, d_model)

        # Stage 1 (Inter-modal)
        self.inter = InterModalStage(d_model)

        # Stage 2 (Intra-modal)
        self.intra = IntraModalStage(d_model)

        # Fusion
        self.fusion = FusionModule(d_model)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, Ya, Yg, Yp):
        # Step 1: Embedding
        Za = self.embed_a(Ya)
        Zg = self.embed_g(Yg)
        Zp = self.embed_p(Yp)

        # Step 2: Inter-modal interaction
        Za, Zg, Zp = self.inter(Za, Zg, Zp)

        # Step 3: Intra-modal refinement
        Za, Zg, Zp = self.intra(Za, Zg, Zp)


        Zg = align_time(Zg, Za.shape[1])
        Zp = align_time(Zp, Za.shape[1])
        # Step 4: Fusion
        Y_fusion = self.fusion(Za, Zg, Zp)
        Y_fusion = self.norm(Y_fusion)
        return Y_fusion
