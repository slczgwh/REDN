import torch
from torch import nn, optim
from .utils import AverageMeter
from tqdm import tqdm
import os
from opennre.model.para_loss import PARALoss
from opennre.framework.f1_metric import F1Metric
import pickle as pk


class SentenceRE(nn.Module):

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 ckpt,
                 max_epoch=100,
                 lr=0.1,
                 weight_decay=1e-5,
                 opt='sgd',
                 add_subject_loss=False,
                 loss_func=PARALoss(),
                 metric=F1Metric()
                 ):

        super().__init__()
        self.metric = metric
        self.add_subject_loss = add_subject_loss
        self.max_epoch = max_epoch
        # Load data
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Model
        self.model = model
        self.parallel_model = nn.DataParallel(self.model)
        # nn.parallel.DistributedDataParallel(self.model,device_ids=)
        # Criterion
        self.loss_func = loss_func
        self.subject_loss = torch.nn.CrossEntropyLoss()
        # Params and optimizer
        params = self.parameters()
        self.lr = lr
        if opt == 'sgd':
            self.optimizer = optim.SGD(params, lr, weight_decay=weight_decay)
        elif opt == 'adam':
            self.optimizer = optim.Adam(params, lr, weight_decay=weight_decay)
        elif opt == 'adamw':  # Optimizer for BERT
            from transformers import AdamW
            params = list(self.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            grouped_params = [
                {
                    'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
                    'weight_decay': 0.01,
                    'lr': lr,
                    'ori_lr': lr
                },
                {
                    'params': [p for n, p in params if any(nd in n for nd in no_decay)],
                    'weight_decay': 0.0,
                    'lr': lr,
                    'ori_lr': lr
                }
            ]
            self.optimizer = AdamW(grouped_params, correct_bias=False)
        else:
            raise Exception("Invalid optimizer. Must be 'sgd' or 'adam' or 'adamw'.")
        # Cuda
        if torch.cuda.is_available():
            self.cuda()
        # Ckpt
        self.ckpt = ckpt

    def train_model(self, warmup=True, metric='acc'):
        best_metric = 0
        global_step = 0

        for epoch in range(self.max_epoch):
            self.train()

            print("=== Epoch %d train ===" % epoch)
            avg_loss = AverageMeter()
            avg_acc = AverageMeter()
            t = tqdm(self.train_loader)

            # self.metric.reset()
            data_idx = 0
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                subject_label = data[-1]
                label = data[-2]
                args = data[0:2]
                logits, subject_label_logits, attx = self.parallel_model(*args)
                loss = self.loss_func(logits, label)

                if self.add_subject_loss:
                    subject_loss = self.subject_loss(subject_label_logits.transpose(1, 2), subject_label)
                    loss += subject_loss

                l = list(logits.detach().cpu().numpy())
                # self.metric.eval(l, self.train_loader.dataset.data[data_idx:data_idx + len(l)])
                data_idx += len(l)

                avg_loss.update(loss.item(), 1)

                t.set_postfix(loss=avg_loss.avg)
                # Optimize
                if warmup == True:
                    warmup_step = 300
                    if global_step < warmup_step:
                        warmup_rate = float(global_step) / warmup_step
                    else:
                        warmup_rate = 1.0
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * warmup_rate
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
            # Val 
            print("=== Epoch %d val ===" % epoch)
            if self.val_loader is not None:
                result = self.eval_model(self.val_loader)
                if result[metric] > best_metric:
                    print("Best ckpt and saved.")
                    folder_path = '/'.join(self.ckpt.split('/')[:-1])
                    if not os.path.exists(folder_path):
                        os.mkdir(folder_path)
                    # torch.save({'state_dict': self.model.state_dict()}, self.ckpt)
                    torch.save(self.parallel_model, self.ckpt)
                    best_metric = result[metric]
            else:
                torch.save(self.parallel_model, self.ckpt)
        print("Best %s on val set: %f" % (metric, best_metric))

    def eval_model(self, eval_loader):
        self.eval()
        # avg_acc = AverageMeter()
        self.metric.reset()
        data_idx = 0
        with torch.no_grad():
            t = tqdm(eval_loader)
            for iter, data in enumerate(t):
                if torch.cuda.is_available():
                    for i in range(len(data)):
                        try:
                            data[i] = data[i].cuda()
                        except:
                            pass
                label = data[-1]
                args = data[0:2]
                logits, _, attx = self.parallel_model(*args)
                l = list(logits.cpu().numpy())
                self.metric.eval(l, eval_loader.dataset.data[data_idx:data_idx + len(l)])
                data_idx += len(l)

                t.set_postfix(f1=self.metric.get_result()["without_na_micro_f1"])
        print(self.metric.get_result())
        return self.metric.get_result()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
