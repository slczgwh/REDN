[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-nyt)](https://paperswithcode.com/sota/relation-extraction-on-nyt?p=downstream-model-design-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=downstream-model-design-of-pre-trained)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/downstream-model-design-of-pre-trained/relation-extraction-on-webnlg)](https://paperswithcode.com/sota/relation-extraction-on-webnlg?p=downstream-model-design-of-pre-trained)
# REDN

This is the prototype code for Relation Extraction Downstream 
Network of pre-trained language model, supporting our paper [*Downstream Model Design of Pre-trained Language Model for Relation Extraction Task*.](https://arxiv.org/abs/2004.03786)

Part of this code are revised based on [Allennlp](https://github.com/allenai/allennlp).

## Datasets

You can get Datasets from [OPENNRE](https://github.com/thunlp/OpenNRE) and [WebNLG 2.0](https://gitlab.com/shimorina/webnlg-dataset/tree/master/release_v2)

## Getting Start

Set your own paths in example/configs.py, including pre-trained model path, root path of data and output name.
Run example/redn_trainer.py with args **dataset** and **mode**. **dataset** can be nyt10, semeval or webnlg.
 **mode** can be t for training and e for evaluation. For example ,to train SemEval, try
 ```
python train.py train -s /your/output/path -f redn/training_configs/redn-{datasset}.jsonnet
```

## Another Branch

We establish a [new branch](https://github.com/slczgwh/REDN/tree/allennlp-based) based on [Allennlp](https://github.com/allenai/allennlp)
. It will be more friendly and powerful while using in real projects, though developers should spend sometime to write dataset-readers.

## Logs

If you are not able to run these codes, you can also check all the logs in ./logs.
