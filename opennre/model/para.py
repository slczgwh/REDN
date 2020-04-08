import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter

from .base_model import SentenceRE


class PARA(SentenceRE):
    """
    Softmax classifier for sentence-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id, num_token_labels, subject_1=False, use_cls=True):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.use_cls = use_cls
        self.subject_1 = subject_1
        self.num_token_labels = num_token_labels
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        hidden_size = self.sentence_encoder.hidden_size
        self.fc = nn.Linear(hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

        self.subject_output_fc = nn.Linear(hidden_size, self.num_token_labels)
        # self.bias = Parameter(torch.zeros((num_class, seq_len, seq_len)))

        self.attn_score = MultiHeadAttention(input_size=hidden_size,
                                             output_size=num_class * hidden_size,
                                             num_heads=num_class)

    def infer(self, item):
        self.eval()
        item = self.sentence_encoder.tokenize(item)
        logits = self.forward(*item)
        logits = self.softmax(logits)
        score, pred = logits.max(-1)
        score = score.item()
        pred = pred.item()
        return self.id2rel[pred], score

    def forward(self, token, att_mask):
        bs = token.shape[0]
        sl = token.shape[1]

        rep, hs, atts = self.sentence_encoder(token, att_mask)  # (B, H)
        if self.subject_1:
            subject_output = hs[-1]  # BS * SL * HS
        else:
            subject_output = hs[-2]  # BS * SL * HS
        subject_output_logits = self.subject_output_fc(subject_output)  # BS * SL * NTL

        if self.use_cls:
            subject_output = subject_output + rep.view(-1, 1, rep.shape[-1])
        # score = self.attn_score(hs[-1], subject_output)  # BS * NTL * SL * SL
        score = self.attn_score(hs[-1], subject_output)  # BS * NTL * SL * SL

        score = score.sigmoid()

        return score, subject_output_logits, atts[-1]


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, output_size, num_heads, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_heads = num_heads
        self.d_model_size = input_size

        self.depth = int(output_size / self.num_heads)

        self.Wq = torch.nn.Linear(input_size, output_size)
        self.Wk = torch.nn.Linear(input_size, output_size)

    def split_into_heads(self, x, batch_size):
        x = x.reshape(batch_size, -1, self.num_heads, self.depth)  # BS * SL * NH * H
        return x.permute([0, 2, 1, 3])  # BS * NH * SL * H

    def forward(self, k, q):  # BS * SL * HS
        batch_size = q.shape[0]

        q = self.Wq(q)  # BS * SL * OUT
        k = self.Wk(k)  # BS * SL * OUT

        # q = F.dropout(q, 0.8, training=self.training)
        # k = F.dropout(k, 0.8, training=self.training)

        q = self.split_into_heads(q, batch_size)  # BS * NH * SL * H
        k = self.split_into_heads(k, batch_size)  # BS * NH * SL * H

        attn_score = torch.matmul(q, k.permute(0, 1, 3, 2))
        attn_score = attn_score / np.sqrt(k.shape[-1])

        # scaled_attention = output[0].permute([0, 2, 1, 3])
        # attn = output[1]
        # original_size_attention = scaled_attention.reshape(batch_size, -1, self.d_model_size)
        # output = self.dense(original_size_attention)

        return attn_score
