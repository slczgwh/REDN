#这里默认指定了中文版本的bert，根据数据需要选择自己的bert版本。也可以换成其他bert的变种，如albert，roberta之类
local model_path = "/mnt/nas1/NLP/pretrained_model/bert/bert_torch/uncased_L-12_H-768_A-12";
# local pretrain_transforms_hidden = 768;  #分类任务这里可以不用设置，取默认值即可
local do_lowercase = true;
local namespace = "token";

#数据文件路径，根据自己的文件路径设置
local train_data_path = "/mnt/nas1/pare/benchmark1/semeval/train.json";
local validation_data_path = "/mnt/nas1/pare/benchmark1/semeval/dev.json";
local test_data_path = "/mnt/nas1/pare/benchmark1/semeval/test.json";


#根据实际情况自行设置
local batch_size = 64;
local num_epochs = 50;
local patience = 3;

{
  "dataset_reader": {
    "type": "relation_extraction_reader",
    "lazy":false,
    "tokenizer":{
      "type":"pretrained_transformer",
      "model_name":model_path,
      "do_lowercase":do_lowercase,
    },
    "token_indexers": {
      "tokens": {
          "type": "bert-pretrained",
          "pretrained_model":model_path,
          "do_lowercase": do_lowercase,
          "use_starting_offsets": true
      },
    }
  },
  "train_data_path": train_data_path,
  "validation_data_path": validation_data_path,
  "test_data_path": test_data_path,
  "evaluate_on_test": true,
  "model": {
    "type": "redn",
    "model_path":model_path,
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens","num_tokens"]],
    "batch_size": batch_size,
  },
  "trainer": {
    "num_epochs": num_epochs,
    "patience": patience,
    "cuda_device": 5,
    "validation_metric": "+micro_f1",
    "num_serialized_models_to_keep":3,
    "optimizer": {
      "type": "adam",
      "lr":3e-5,
      "weight_decay":1e-5,
    }

  }
}