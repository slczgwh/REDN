import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from .base_model import SentenceRE


class PARALossSoftmax(nn.Module):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()

    def forward(self, score, predicate_one_hot_labels):
        # bs * nl * sl * sl
        soft = True
        if predicate_one_hot_labels.is_sparse:
            predicate_one_hot_labels = predicate_one_hot_labels.to_dense()
        if not soft:
            entity_mask = predicate_one_hot_labels.sum(dim=1)  # .repeat_interleave(score.shape[1], dim=1).float()

            label = predicate_one_hot_labels.argmax(dim=1)  # BS, SL, SL

            loss = F.cross_entropy(score, label, reduction="none")  # BS, SL, SL
            loss = loss * entity_mask  # BS, SL, SL
            loss = loss.sum(dim=(1, 2)) / entity_mask.sum(dim=(1, 2))  # BS
            loss = loss.mean()  # 1
        else:
            entity_mask = predicate_one_hot_labels.sum(dim=1, keepdim=True).repeat_interleave(score.shape[1],
                                                                                              dim=1).float()  # BS, NL, SL, SL
            score = (score * entity_mask).sum(dim=(2, 3))  # BS, NL
            label = predicate_one_hot_labels.sum(dim=(2, 3)).argmax(dim=-1)  # BS
            loss = F.cross_entropy(score, label)  # 1

        return loss
