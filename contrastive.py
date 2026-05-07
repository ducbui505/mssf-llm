"""
contrastive.py – Supervised Contrastive Loss with edge-case guards.

Kéo các mẫu cùng frequency class lại gần nhau (trong embedding space),
đẩy các mẫu khác class ra xa.  Giải quyết class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features : (batch, d_model) – e.g. f_fused from cross-modal fusion
            labels   : (batch,)         – class indices 0..4

        Returns:
            Scalar loss.  Returns 0. if batch has no valid positive pairs.
        """
        device = features.device
        batch_size = features.size(0)

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 normalize → cosine similarity
        features = F.normalize(features, dim=1)

        # Similarity matrix (batch × batch)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Masks
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        mask_pos = (labels_row == labels_col).float()                   # same class
        mask_self = torch.eye(batch_size, device=device)
        mask_pos = mask_pos - mask_self                                  # exclude self

        # Count positives per sample
        n_pos = mask_pos.sum(dim=1)  # (B,)

        # Guard: skip samples whose class has only 1 sample in this batch
        valid = n_pos > 0  # (B,) bool
        if not valid.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability: subtract max per row
        logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
        logits = sim_matrix - logits_max.detach()

        # Denominator: sum over all j ≠ i
        exp_logits = torch.exp(logits) * (1.0 - mask_self)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean log-prob over positive pairs (only for valid samples)
        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / n_pos.clamp(min=1)

        loss = -mean_log_prob[valid].mean()
        return loss
