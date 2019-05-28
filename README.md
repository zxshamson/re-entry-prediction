# Introduction:
This is the implementation in PyTorch for my ACL paper:

"Joint Effects of Context and User History for Predicting Online Conversation Re-entries"


# Requirement:

* Python: 2.7+

* Pytorch: 0.4+

* Sklearn: 0.20.0

# Usage:

`python train.py [filename] [modelname]`

```
[filename]: "twitter.data" or "reddit.data".
[modelname]: "AVG", "CNN", "LSTM", "LSTMMEM", "LSTMBiA", "LSTMBiANa", "LSTMMerge", "LSTMAtt".

optional arguments:
  --cuda_dev          choose to use which GPU (default: "0")
  --max_word_num      max word number in vocabulary, -1 means all words appear in dataset (default: -1)
  --entrytime         given how many entries as context  (default: 1)
  --train_pc          percentage of conversations for training (default: 0.8)
  --embedding_dim     dimension for word embedding (default: 200)
  --hidden_dim        dimension for hidden states (default: 200)
  --kernal_num        number of kernals for CNN encoder (default: 50)
  --hop_num           number of hops for MEM network (default: 3)
  --batch_size        batch size during training (default: 8)
  --max_epoch         maximum iteration times (default: 200)
  --history_size      maximum number of messages for user history (default:50)
  --lr                learning rate during training (default: 0.001)
  --dropout           dropout rate (default: 0.2)
  --num_layer         number of layers for LSTM (default: 1)
  --model_num_layer   number of layers for LSTM in bi-attention mechanism (default: 2)
  --need_history      whether current model needs history (action="store_true")
  --bi_direction      whether change LSTM to BiLSTM (action="store_true")
  --runtime           record the current running time (default: 0)
  --train_weight      weights in loss function during training (default: 2)
```

# Datasets:

format in each line:

[Conv ID] \t [Msg ID] \t [Parent ID] \t [Original sentence] \t [words after preprocessing] \t [User ID] \t [posting time]

(twitter dataset doesn't have time infos, but the conversations are ordered by posting time)
