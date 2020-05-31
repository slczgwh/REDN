# bert config
local model_path = "/mnt/nas1/NLP/pretrained_model/bert/bert_torch/uncased_L-12_H-768_A-12";
local do_lowercase = true;
local namespace = "token";

# data path
local train_data_path = "/mnt/nas1/pare/benchmark1/semeval/train.json";
local validation_data_path = "/mnt/nas1/pare/benchmark1/semeval/dev.json";
local test_data_path = "/mnt/nas1/pare/benchmark1/semeval/test.json";

# training config
local batch_size = 20;
local num_epochs = 100;
local patience = 0;

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