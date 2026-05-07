"""
cross_modal.py – Two cross-modal fusion variants.

Variant A (default): Gated Fusion
Variant B (ablation): Multi-head Cross-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ═══════════════════════════════════════════════════════════════
#  Variant A – Gated Fusion (recommended default)
# ═══════════════════════════════════════════════════════════════

class CrossModalGatedFusion(nn.Module):
    """
    Learns a soft gate to blend structured MSSF features with
    projected LLM features.  Includes a residual connection so
    that when the gate is ~0 the output ≈ f_struct (safe init).

        gate   = σ( W_g · [f_struct ; f_llm] + b_g )
        f_fused = gate * f_struct + (1 − gate) * W_v(f_llm) + f_struct   # residual
    """

    def __init__(self, d_structured: int = 384, d_llm: int = 384, d_model: int = 384):
        super().__init__()
        self.gate_proj = nn.Linear(d_structured + d_llm, d_model)
        self.value_proj = nn.Linear(d_llm, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Init gate bias negative so sigmoid ≈ 0 at start → behaves like baseline
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, f_structured: torch.Tensor, f_llm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_structured : (batch, d_structured)  – after MSSF self-attention
            f_llm        : (batch, d_llm)         – from LLMBranch
        Returns:
            f_fused      : (batch, d_model)
        """
        gate = torch.sigmoid(self.gate_proj(torch.cat([f_structured, f_llm], dim=1)))
        v_llm = self.value_proj(f_llm)
        fused = gate * f_structured + (1.0 - gate) * v_llm
        return self.layer_norm(fused + f_structured)  # residual


# ═══════════════════════════════════════════════════════════════
#  Variant B – Multi-head Cross-Attention (ablation)
# ═══════════════════════════════════════════════════════════════

class CrossModalMultiHeadAttention(nn.Module):
    """
    Treats the 3 encoder outputs (each 128-d) as a length-3 sequence
    for Q, and uses the LLM feature as a single K/V token.

    This gives the model the ability to weight how much each encoder
    stream should attend to text information.

    Input f_struct is expected to be (batch, 384) and is reshaped
    internally to (batch, 3, 128).
    """

    def __init__(
        self,
        d_structured: int = 384,
        d_llm: int = 384,
        d_model: int = 384,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_encoders = 3
        self.d_per_encoder = d_structured // self.n_encoders  # 128
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(self.d_per_encoder, d_model)
        self.W_K = nn.Linear(d_llm, d_model)
        self.W_V = nn.Linear(d_llm, d_model)
        self.out_proj = nn.Linear(d_model, self.d_per_encoder)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_structured)

    def forward(self, f_structured: torch.Tensor, f_llm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_structured : (batch, 384) – concatenated 3×128 encoder features
            f_llm        : (batch, d_llm)
        Returns:
            f_fused      : (batch, 384)
        """
        B = f_structured.size(0)

        # Reshape: (B, 384) → (B, 3, 128)
        seq = f_structured.view(B, self.n_encoders, self.d_per_encoder)

        # Q from structured sequence: (B, 3, d_model)
        Q = self.W_Q(seq)
        # K, V from LLM (single token): (B, 1, d_model)
        K = self.W_K(f_llm).unsqueeze(1)
        V = self.W_V(f_llm).unsqueeze(1)

        # Reshape for multi-head: (B, n_heads, seq_len, d_k)
        Q = Q.view(B, self.n_encoders, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, 1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, 1, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: (B, n_heads, 3, 1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        # Attended values: (B, n_heads, 3, d_k)
        attended = torch.matmul(weights, V)

        # Merge heads: (B, 3, d_model)
        attended = attended.transpose(1, 2).contiguous().view(B, self.n_encoders, self.d_model)

        # Project back: (B, 3, 128)
        attended = self.out_proj(attended)

        # Flatten + residual: (B, 384)
        out = attended.view(B, -1)
        return self.layer_norm(out + f_structured)
