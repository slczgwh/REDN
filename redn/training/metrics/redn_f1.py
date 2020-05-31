from typing import List, Optional, Union

import torch
from overrides import overrides

from allennlp.training.metrics.metric import Metric

CORRECT = "correct"
TOTAL = "total"
PRED_POSITIVE = "pred_positive"
GOLD_POSITIVE = "gold_positive"


@Metric.register("redn_f1")
class F1Metric(Metric):

    def __init__(self, multi_label=True, na_id=None, ignore_na=False, print_error_prob=0, rel2id=None):
        self.print_error_prob = print_error_prob
        self.multi_label = multi_label
        self.na_id = na_id
        self.ignore_na = ignore_na
        self.id2rel = None
        if rel2id is not None:
            self.id2rel = {value: key for key, value in rel2id.items()}
        self.reset()

    @overrides
    def __call__(self,
                 pred: torch.Tensor,
                 label: torch.Tensor,
                 mask: Optional[torch.Tensor] = None):

        if self.na_id:
            # 当na预测为正时，将所有非na类别预测值降低0.5
            pred = pred - ((pred[:, self.na_id] > 0.5).unsqueeze(1).float() / 2)
        # pred = pred.scatter_add_(1, pred.new_full((pred.shape[0], 1), self.na_id, dtype=torch.long), (pred[:, self.na_id] > 0.5).unsqueeze(1).float() / 2)
        # 当所有类别都预测为负时，将na的预测值增加0.5
        # pred = pred.scatter_add_(1, pred.new_full((pred.shape[0], 1), self.na_id, dtype=torch.long), (pred < 0.5).all(dim=1, keepdim=True).float() / 2)

        pred = pred > 0.5
        one_hot = pred.new_zeros(pred.shape).scatter_(1, label.unsqueeze(dim=-1), 1)

        if self.na_id:
            s_mask = (label != self.na_id).unsqueeze(dim=1)
            pred = pred * s_mask
            one_hot = one_hot * s_mask

        if self.ignore_na:
            mask = pred.new_ones(pred.shape)
            mask[:, self.na_id] = 0
            pred = pred * mask
            one_hot = one_hot * mask

        self.res[PRED_POSITIVE] += pred.sum().item()
        self.res[GOLD_POSITIVE] += one_hot.sum().item()
        self.res[CORRECT] += ((pred == one_hot) * (1 == one_hot)).sum().item()
        self.res[TOTAL] += pred.shape[0]

    @staticmethod
    def _get_result(res):
        acc = float(res[CORRECT]) / float(res[TOTAL] + 1e-9)
        micro_p = float(res[CORRECT]) / float(res[PRED_POSITIVE] + 1e-9)
        micro_r = float(res[CORRECT]) / float(res[GOLD_POSITIVE] + 1e-9)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)
        return {'micro_f1': micro_f1, 'micro_p': micro_p, 'micro_r': micro_r, 'acc': acc}

    def get_metric(self, reset: bool):
        res = F1Metric._get_result(self.res)
        # res["without_na_res"] = F1Metric._get_result(self.without_na_res)
        # res["na_res"] = F1Metric._get_result(self.na_res)
        # res["without_na_micro_f1"] = res["without_na_res"]["micro_f1"]
        # res["normal"] = F1Metric._get_result(self.normal_res)
        # res["over_lapping"] = F1Metric._get_result(self.over_lapping_res)
        # res["multi_label"] = F1Metric._get_result(self.multi_label_res)

        # res["triple_res"] = {}
        # for i in range(len(self.triple_count_res)):
        #     res["triple_res"][str(i)] = F1Metric._get_result(self.triple_count_res[i])

        if reset:
            self.reset()

        return res

    def reset(self):
        self.res = {CORRECT: 0, TOTAL: 0, PRED_POSITIVE: 0, GOLD_POSITIVE: 0}

        # self.without_na_res = self.res.copy()
        # self.na_res = self.res.copy()
        # self.normal_res = self.res.copy()
        # self.over_lapping_res = self.res.copy()
        # self.multi_label_res = self.res.copy()
        #
        # self.triple_count_res = [self.res.copy() for _ in range(4)]
        # self.df = pd.DataFrame(columns=["Text", "Subject", "Object", "Gold Label", "Predict"])

