import torch
from opennre.dataset.utils import is_normal, is_multi_label, is_over_lapping
from opennre.framework.data_loader import SentenceREDataset
import random
import pandas as pd

CORRECT = "correct"
TOTAL = "total"
PRED_POSITIVE = "pred_positive"
GOLD_POSITIVE = "gold_positive"


class F1Metric(object):
    def __init__(self, multi_label=True, na_id=-1, ignore_na=False, print_error_prob=0, rel2id=None):
        self.print_error_prob = print_error_prob
        self.multi_label = multi_label
        self.na_id = na_id
        self.ignore_na = ignore_na
        self.id2rel = None
        if rel2id is not None:
            self.id2rel = {value: key for key, value in rel2id.items()}
        self.reset()

    def eval(self, pred_result, data_list: list):
        pred_result = pred_result  # B*NL*SL*SL
        for i, d in enumerate(data_list):
            is_print = random.random() < self.print_error_prob
            epl_c = d["entity_pair_list"].copy()
            epl_c = [ep for ep in epl_c if ep[2] != self.na_id]

            is_normal_data = is_normal(epl_c)
            is_multi_label_data = is_multi_label(epl_c)
            is_over_lapping_data = is_over_lapping(epl_c)
            triple_count = len(epl_c)
            triple_count = min(triple_count, len(self.triple_count_res) - 1)

            checked_epl_id = []
            epl = d["entity_pair_list"]
            for e_idx, ep in enumerate(epl):
                if e_idx in checked_epl_id:
                    continue
                checked_epl_id.append(e_idx)
                pr = torch.from_numpy(pred_result[i])
                em = SentenceREDataset.get_entity_mask(d["entity_list"][ep[0]],
                                                       d["entity_list"][ep[1]],
                                                       d["new_index"],
                                                       pr.shape[-1]).to(pr.device)  # SL*SL
                gold_label = [ep[2]]
                for _e_idx, _ep in enumerate(epl):
                    if _e_idx in checked_epl_id:
                        continue
                    if _ep[0] == ep[0] and _ep[1] == ep[1]:
                        gold_label.append(_ep[2])
                        checked_epl_id.append(_e_idx)

                _res = ((pr * em).sum(dim=(1, 2)) / em.sum()).cpu().numpy()  # NL
                res = [0] * len(_res)
                if self.multi_label:
                    if self.na_id > -1 and _res[self.na_id] > 0.5:
                        res[self.na_id] = 1
                    else:
                        # res[self.na_id] = 0
                        res = (_res > 0.5).astype(int)  # NL
                else:
                    res = (res == max(_res)).astype(int)
                if not (self.ignore_na and ep[2] == self.na_id):
                    gold_count = len(gold_label)
                    for idx, s in enumerate(res):
                        if idx in gold_label and s == 1:
                            self.res[CORRECT] += 1
                            self.triple_count_res[triple_count][CORRECT] += 1 if idx != self.na_id else 0
                            self.without_na_res[CORRECT] += 1 if idx != self.na_id else 0
                            self.na_res[CORRECT] += 1 if idx == self.na_id else 0
                            self.normal_res[CORRECT] += 1 if is_normal_data and idx != self.na_id else 0
                            self.multi_label_res[CORRECT] += 1 if is_multi_label_data and idx != self.na_id else 0
                            self.over_lapping_res[CORRECT] += 1 if is_over_lapping_data and idx != self.na_id else 0
                            gold_count -= 1
                        if idx in gold_label:
                            self.res[GOLD_POSITIVE] += 1
                            self.triple_count_res[triple_count][GOLD_POSITIVE] += 1 if idx != self.na_id else 0
                            self.without_na_res[GOLD_POSITIVE] += 1 if idx != self.na_id else 0
                            self.na_res[GOLD_POSITIVE] += 1 if idx == self.na_id else 0
                            self.normal_res[GOLD_POSITIVE] += 1 if is_normal_data and idx != self.na_id else 0
                            self.multi_label_res[GOLD_POSITIVE] += 1 if is_multi_label_data and idx != self.na_id else 0
                            self.over_lapping_res[GOLD_POSITIVE] += 1 if is_over_lapping_data and idx != self.na_id else 0
                        if s == 1:
                            self.res[PRED_POSITIVE] += 1
                            self.triple_count_res[triple_count][PRED_POSITIVE] += 1 if idx != self.na_id else 0
                            self.without_na_res[PRED_POSITIVE] += 1 if idx != self.na_id else 0
                            self.na_res[PRED_POSITIVE] += 1 if idx == self.na_id else 0
                            self.normal_res[PRED_POSITIVE] += 1 if is_normal_data and idx != self.na_id else 0
                            self.multi_label_res[PRED_POSITIVE] += 1 if is_multi_label_data and idx != self.na_id else 0
                            self.over_lapping_res[PRED_POSITIVE] += 1 if is_over_lapping_data and idx != self.na_id else 0
                    self.res[TOTAL] += 1
                    self.triple_count_res[triple_count][TOTAL] += 1 if ep[2] != self.na_id else 0
                    self.without_na_res[TOTAL] += 1 if ep[2] != self.na_id else 0
                    self.na_res[TOTAL] += 1 if ep[2] == self.na_id else 0
                    self.normal_res[TOTAL] += 1 if is_normal_data and ep[2] != self.na_id else 0
                    self.multi_label_res[TOTAL] += 1 if is_multi_label_data and ep[2] != self.na_id else 0
                    self.over_lapping_res[TOTAL] += 1 if is_over_lapping_data and ep[2] != self.na_id else 0

                    if is_print and gold_count != 0 and len(epl) > 1:
                        gold_label_str = [g if self.id2rel is None else self.id2rel[g] for g in gold_label]
                        pred_str = [idx if self.id2rel is None else self.id2rel[idx] for idx, s in enumerate(res) if s == 1]

                        self.df = self.df.append({"Text": " ".join(d["token"]),
                                                  "Subject": d["entity_list"][ep[0]],
                                                  "Object": d["entity_list"][ep[1]],
                                                  "Gold Label": ",".join(gold_label_str),
                                                  "Predict": ",".join(pred_str),
                                                  }, ignore_index=True, )
                        print(" ")
                        print("-------------------------------------------------------")
                        print("sentence is:")
                        print(" ".join(d["token"]))
                        print("subject :%s" % d["entity_list"][ep[0]])
                        print("object :%s" % d["entity_list"][ep[1]])
                        for g in gold_label:
                            print("gold label :%s" % (g if self.id2rel is None else self.id2rel[g]))
                        for idx, s in enumerate(res):
                            if s == 1:
                                print("pred label :%s" % (idx if self.id2rel is None else self.id2rel[idx]))
                        print("-------------------------------------------------------")

    @staticmethod
    def _get_result(res):
        acc = float(res[CORRECT]) / float(res[TOTAL] + 1e-9)
        micro_p = float(res[CORRECT]) / float(res[PRED_POSITIVE] + 1e-9)
        micro_r = float(res[CORRECT]) / float(res[GOLD_POSITIVE] + 1e-9)
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r + 1e-9)
        return {'micro_f1': micro_f1, 'micro_p': micro_p, 'micro_r': micro_r, 'acc': acc}

    def get_result(self):
        res = F1Metric._get_result(self.res)
        res["without_na_res"] = F1Metric._get_result(self.without_na_res)
        res["na_res"] = F1Metric._get_result(self.na_res)
        res["without_na_micro_f1"] = res["without_na_res"]["micro_f1"]
        res["normal"] = F1Metric._get_result(self.normal_res)
        res["over_lapping"] = F1Metric._get_result(self.over_lapping_res)
        res["multi_label"] = F1Metric._get_result(self.multi_label_res)

        res["triple_res"] = {}
        for i in range(len(self.triple_count_res)):
            res["triple_res"][str(i)] = F1Metric._get_result(self.triple_count_res[i])

        return res

    def reset(self):
        self.res = {CORRECT: 0, TOTAL: 0, PRED_POSITIVE: 0, GOLD_POSITIVE: 0}

        self.without_na_res = self.res.copy()
        self.na_res = self.res.copy()
        self.normal_res = self.res.copy()
        self.over_lapping_res = self.res.copy()
        self.multi_label_res = self.res.copy()

        self.triple_count_res = [self.res.copy() for _ in range(4)]
        self.df = pd.DataFrame(columns=["Text", "Subject", "Object", "Gold Label", "Predict"])
