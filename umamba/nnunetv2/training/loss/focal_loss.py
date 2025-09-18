import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """多分类Focal Loss, 适配nnUNet格式 (logits: B,C,*, target: B,1,*)"""
    def __init__(self, gamma=2.0, alpha=0.25, ignore_index=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, target):
        # target shape B,1,* -> squeeze
        if target.dim() == logits.dim():
            # assume already sparse indices broadcasted incorrectly; try argmax
            if target.size(1) == logits.size(1):
                target = target.argmax(1, keepdim=True)
        if target.size(1) == 1:
            target = target[:,0]
        # logits: B,C, ... ; flatten batch + spatial
        B, C = logits.shape[:2]
        logits_flat = logits.reshape(B, C, -1)
        target_flat = target.reshape(B, -1)
        # mask ignore
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
        else:
            mask = torch.ones_like(target_flat, dtype=torch.bool)
        # gather logits for true class
        log_probs = F.log_softmax(logits_flat, dim=1)
        probs = log_probs.exp()
        gather_idx = target_flat.clamp(min=0)  # prevent negative index
        true_logp = log_probs.gather(1, gather_idx.unsqueeze(1)).squeeze(1)
        true_p = probs.gather(1, gather_idx.unsqueeze(1)).squeeze(1)
        # alpha weighting per class (scalar alpha -> pos class weight, else treat as scalar)
        alpha = self.alpha
        focal_weight = (1 - true_p) ** self.gamma
        loss = -alpha * focal_weight * true_logp
        loss = loss * mask
        if self.reduction == 'mean':
            denom = mask.sum().clamp(min=1)
            return loss.sum() / denom
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss.reshape(B, -1)

