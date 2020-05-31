import torch
import torch.nn as nn

from typing import Dict

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import SpanBasedF1Measure, FBetaMeasure

from redn.training.metrics.redn_f1 import F1Metric
from redn.models.redn.redn_loss import RednLoss
from redn.models.redn.multi_head_attn_score import MultiHeadAttentionScore

from pytorch_transformers import BertModel


@Model.register("redn_with_ner")
class RednWithNER(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 model_path: str,
                 do_ner_task=False,
                 subject_1=True,
                 use_cls=True,
                 ):
        super().__init__(vocab)
        self.bert = BertModel.from_pretrained(model_path, output_hidden_states=True)
        self.hidden_size = self.bert.config.hidden_size

        self.loss_fn = RednLoss()
        try:
            self.naid = self.vocab.get_token_index("Other", "labels")
        except KeyError:
            self.naid = None
        self.f1 = F1Metric(na_id=self.naid, ignore_na=(self.naid is not None))
        label_list = []
        for k, v in self.vocab._index_to_token["ner_tags"].items():
            if v != "O":
                label_list.append(k)
        self.ner_metric = FBetaMeasure(labels=label_list)

        self.subject_1 = subject_1
        self.use_cls = use_cls
        self.do_ner_task = do_ner_task

        self.num_rel_class = self.vocab.get_vocab_size("labels")
        self.attn_score = MultiHeadAttentionScore(input_size=self.hidden_size,
                                                  output_size=self.num_rel_class * self.hidden_size,
                                                  num_heads=self.num_rel_class)

        self.ner_loss = torch.nn.CrossEntropyLoss()
        self.num_ner_tags = self.vocab.get_vocab_size("ner_tags")
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2, bias=True, batch_first=True, dropout=0, bidirectional=True)
        self.ner_liner = nn.Linear(in_features=self.hidden_size * self.lstm.num_layers, out_features=self.num_ner_tags)

    def forward(  # type: ignore
            self,
            tokens,
            span_h=None,
            span_t=None,
            rel_label: torch.IntTensor = None,
            tags=None,
    ) -> Dict[str, torch.Tensor]:
        token_ids = tokens["tokens"]
        tf = tokens["tokens-offsets"]

        _, rep, hs = self.bert(token_ids)

        subject_output = hs[-1] if self.subject_1 else hs[-2]  # BS, SL, HS

        if self.use_cls:
            subject_output = subject_output + rep.view(-1, 1, rep.shape[-1])

        res = {}
        score = self.attn_score(hs[-1], subject_output).sigmoid()  # BS, NR, SL, SL
        res["logits"] = score
        res["token-offset"] = tf

        if span_h is not None and span_t is not None:
            adj = []
            seq_len = token_ids.shape[1]
            for i in range(len(token_ids)):
                h, t = token_ids.new_zeros((seq_len), dtype=torch.int8), token_ids.new_zeros((seq_len), dtype=torch.int8)
                h[tf[i][span_h[i][0]] if span_h[i][0] + 1 < len(tf[i]) else tokens["mask"][i].sum().item()-1:
                  tf[i][span_h[i][1] + 1] if span_h[i][1] + 1 < len(tf[i]) else tokens["mask"][i].sum().item()-1] = 1
                t[tf[i][span_t[i][0]] if span_t[i][0] + 1 < len(tf[i]) else tokens["mask"][i].sum().item()-1:
                  tf[i][span_t[i][1] + 1] if span_t[i][1] + 1 < len(tf[i]) else tokens["mask"][i].sum().item()-1] = 1
                h, t = h.unsqueeze(dim=0).repeat_interleave(seq_len, dim=0), t.unsqueeze(dim=0).repeat_interleave(seq_len, dim=0)
                adj.append((h + t.t() == 2))
            adj = torch.stack(adj, dim=0).bool()

            pred = (score * adj.unsqueeze(dim=1).repeat(1, score.shape[1], 1, 1).float()).sum(dim=(2, 3)) / adj.sum(dim=(1, 2)).unsqueeze(dim=1).float()
            res["pred"] = pred
            if rel_label is not None:
                res["redn_loss"] = self.loss_fn(score, adj, rel_label)
                self.f1(pred, rel_label)

        ner_res = self.ner_liner(self.lstm(hs[-1])[0])  # BS, SL, NT
        index = tf.unsqueeze(dim=2).repeat(1, 1, self.num_ner_tags)
        ner_res = torch.gather(ner_res, dim=1, index=index)
        res["ner_logits"] = ner_res
        res["ner_pred"] = ner_res.argmax(dim=-1)
        if tags is not None:
            res["ner_loss"] = self.ner_loss(input=ner_res.reshape(-1, self.num_ner_tags), target=tags.reshape(-1))
            self.ner_metric(ner_res, tags)
            res["loss"] = res["ner_loss"] + res["redn_loss"]
        return res

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = self.f1.get_metric(reset)
        ner_metrics = self.ner_metric.get_metric(reset)
        ner_metrics = {"ner_" + k: sum(v) / (len(v)+1e-9) for k, v in ner_metrics.items()}
        metrics.update(ner_metrics)

        return metrics
