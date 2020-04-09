# REDN

This is the prototype code for Relation Extraction Downstream 
Network of pre-trained language model, supporting our paper [*Downstream Model Design of Pre-trained Language Model for Relation Extraction Task*.](https://arxiv.org/abs/2004.03786)

Part of this code are revised based on [OPENNRE](https://github.com/thunlp/OpenNRE).

## Datasets

You can get Datasets from [OPENNRE](https://github.com/thunlp/OpenNRE) and [WebNLG 2.0](https://gitlab.com/shimorina/webnlg-dataset/tree/master/release_v2)

## Getting Start

Set your own paths in example/configs.py, including pre-trained model path, root path of data and output name.
Run example/redn_trainer.py with args **dataset** and **mode**. **dataset** can be nyt10, semeval or webnlg.
 **mode** can be t for training and e for evaluation. For example ,to train SemEval, try
 ```
python redn_trainer semeval t
```

## Logs

If you are not able to run these codes, you can also check all the logs in ./logs.
