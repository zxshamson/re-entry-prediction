import random
import torch
import numpy as np
import torch.utils.data as data
from itertools import chain
from corpus import Corpus


def make_vector(texts, text_size, sent_len):  # Pad the conv/history with 0s to fixed size
    text_vec = []
    for one_text in texts:
        t = []
        for sent in one_text:
            pad_len = max(0, sent_len - len(sent))
            t.append(sent + [0] * pad_len)
        pad_size = max(0, text_size - len(t))
        t.extend([[0] * sent_len] * pad_size)
        text_vec.append(t)
    return torch.LongTensor(text_vec)


class MyDataset(data.Dataset):
    def __init__(self, raw_corpus, conv_ids, part, need_user_history=False, history_size=50, entrytime=1):
        self.need_user_history = need_user_history
        self.history_size = min(history_size, max([len(raw_corpus.user_history[u]) for u in raw_corpus.user_history.keys()]))
        self.data_conv = []
        self.data_label = []
        if self.need_user_history:
            self.data_history = []
        for i in range(part[0], part[1]):
            cid = conv_ids[i]
            users = set()  # Store the users that have participated in the current conv
            for turn in raw_corpus.convs[cid]:
                if turn[0] != -1:
                    users.add(turn[0])
            for u in users:
                c = []
                # Change the userID to 1 if the current turn is given by current user, otherwise 0
                find_history = entrytime
                re_entry = 0
                last_entry = []
                for turn in raw_corpus.convs[cid]:
                    if find_history:
                        if turn[0] == u:
                            turn[0] = 1
                            find_history -= 1
                            if not find_history:
                                last_entry.extend(turn[3:])
                        else:
                            turn[0] = 0
                        c.append(turn)
                    else:
                        if turn[0] == u:
                            re_entry = 1
                if find_history:
                    continue
                self.data_conv.append(c)
                self.data_label.append(re_entry)
                if self.need_user_history:  # Record the latest history_size of history before user's first entry
                    for h in range(len(raw_corpus.user_history[u])):
                        if raw_corpus.user_history[u][h] == last_entry:
                            break
                    current_history = raw_corpus.user_history[u][:h+1]
                    self.data_history.append(current_history[::-1][:self.history_size][::-1])

        self.conv_turn_size = max([len(c) for c in self.data_conv])
        self.conv_sent_len = max([len(sent) for sent in chain.from_iterable([c for c in self.data_conv])])
        self.data_conv = make_vector(self.data_conv, self.conv_turn_size, self.conv_sent_len)
        if self.need_user_history:
            self.history_sent_len = max([len(sent) for sent in chain.from_iterable([h for h in self.data_history])])
            self.data_history = make_vector(self.data_history, self.history_size, self.history_sent_len)
        self.data_label = torch.Tensor(self.data_label)

    def __getitem__(self, idx):
        if self.need_user_history:
            return self.data_conv[idx], self.data_history[idx], self.data_label[idx]
        else:
            return self.data_conv[idx], self.data_label[idx]

    def __len__(self):
        return len(self.data_label)


def form_dataset(raw_corpus, train_percentage, batch_size, need_user_history=False, history_size=50, entrytime=1):
    train_num = int(raw_corpus.convNum * train_percentage)  # Threshold for training data
    test_num = train_num + int(raw_corpus.convNum * ((1.0 - train_percentage) / 2.0))  # Threshold for test data
    train_part = (0, train_num)
    test_part = (train_num, test_num)
    dev_part = (test_num, raw_corpus.convNum)
    conv_ids = raw_corpus.convs.keys()
    random.shuffle(conv_ids)

    train_data = MyDataset(raw_corpus, conv_ids, train_part, need_user_history, history_size, entrytime)
    train_loader = data.DataLoader(train_data, batch_size=batch_size, num_workers=1, shuffle=True)
    test_data = MyDataset(raw_corpus, conv_ids, test_part, need_user_history, history_size, entrytime)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, num_workers=1)
    dev_data = MyDataset(raw_corpus, conv_ids, dev_part, need_user_history, history_size, entrytime)
    dev_loader = data.DataLoader(dev_data, batch_size=batch_size, num_workers=1)

    return train_loader, test_loader, dev_loader


def create_embedding_matrix(filename, corp, embedding_dim=200):
    pretrain_file = 'glove.twitter.27B.200d.txt' if filename[0] == 't' else 'glove.6B.200d.txt'
    pretrain_words = {}
    with open(pretrain_file, 'r') as f:
        for line in f:
            infos = line.split()
            wd = infos[0]
            vec = np.array(infos[1:]).astype(np.float)
            pretrain_words[wd] = vec
    word_idx = corp.r_wordIDs
    vocab_num = corp.wordNum
    weights_matrix = np.zeros((vocab_num, embedding_dim))
    for idx in word_idx.keys():
        try:
            weights_matrix[idx] = pretrain_words[word_idx[idx]]
        except KeyError:
            weights_matrix[idx] = np.random.normal(size=(embedding_dim,))
    if torch.cuda.is_available():  # run in GPU
        return torch.Tensor(weights_matrix).cuda()
    else:
        return torch.Tensor(weights_matrix)
