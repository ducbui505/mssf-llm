"""
llm_branch.py – PubMedBERT feature projection branch.

Nhận vector PubMedBERT đã encode sẵn (768-d) của drug và SE,
kết hợp và chiếu xuống d_model = 384 để khớp với MSSF fusion dim.
"""

import torch
import torch.nn as nn


class LLMBranch(nn.Module):

    def __init__(self, input_dim: int = 768, output_dim: int = 384, dropout: float = 0.1):
        super().__init__()
        hidden = (input_dim + output_dim) // 2  # 576
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim),
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        drug_vec: torch.Tensor,
        se_vec: torch.Tensor,
        drug_mask: torch.Tensor | None = None,
        se_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            drug_vec  : (batch, 768) – PubMedBERT embedding of drug
            se_vec    : (batch, 768) – PubMedBERT embedding of SE
            drug_mask : (batch,) – 1.0 if drug has real text, 0.0 otherwise
            se_mask   : (batch,) – 1.0 if SE has real text, 0.0 otherwise

        Returns:
            (batch, output_dim) – projected LLM features
        """
        # Zero out vectors that have no real text (encoded from empty string)
        if drug_mask is not None:
            drug_vec = drug_vec * drug_mask.unsqueeze(1)
        if se_mask is not None:
            se_vec = se_vec * se_mask.unsqueeze(1)

        combined = drug_vec + se_vec          # (batch, 768)
        out = self.projection(combined)       # (batch, output_dim)
        return self.layer_norm(out)
