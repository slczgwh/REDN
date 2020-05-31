import torch
import torch.nn as nn
import torch.nn.functional as F


class RednLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, score, adj: torch.Tensor, label: torch.Tensor):
        # entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1], dim=1)
        entity_mask = adj.unsqueeze(dim=1).repeat(1, score.shape[1], 1, 1).float()

        entity_sum = adj.sum(dim=(1, 2)).unsqueeze(dim=1).repeat(1, score.shape[1]).float()  # BS, NL

        pohl_mask = adj.sum(dim=(1, 2)) > 0

        pohl = adj.new_zeros(score.shape)
        s_index = label.unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3)
        s_index = s_index.repeat(1, 1, adj.shape[1], adj.shape[1])
        pohl = pohl.scatter(1, s_index, adj.unsqueeze(dim=1)).float()

        loss = ((F.binary_cross_entropy(score, pohl, reduction="none") * entity_mask).sum(dim=(2, 3))[pohl_mask] / entity_sum[pohl_mask]).mean()
        return loss
