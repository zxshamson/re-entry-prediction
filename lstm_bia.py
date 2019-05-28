import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class SentLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, bi_direction=True):
        super(SentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.sent_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bi_direction)
        self.sent_hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPrU
            return (torch.randn(bi, batch_size, self.hidden_dim).cuda(),
                    torch.randn(bi, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(bi, batch_size, self.hidden_dim),
                    torch.randn(bi, batch_size, self.hidden_dim))

    def forward(self, sentences, sent_lens):
        self.sent_hidden = self.init_hidden(len(sentences))
        sorted_sent_lens, indices = torch.sort(sent_lens, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_sentences = sentences[indices]
        packed_sentences = rnn_utils.pack_padded_sequence(sorted_sentences, sorted_sent_lens, batch_first=True)
        lstm_out, self.sent_hidden = self.sent_lstm(packed_sentences, self.sent_hidden)
        if self.bi_direction:
            sent_reps = torch.cat([self.sent_hidden[0][-2], self.sent_hidden[0][-1]], dim=1)
            sent_reps = sent_reps[desorted_indices]
        else:
            sent_reps = self.sent_hidden[0][-1][desorted_indices]

        return sent_reps


class LSTMBiA(nn.Module):
    def __init__(self, embedding_dim, vocab_num, hidden_dim, batch_size, dropout, num_layer, model_num_layer, bi_direction, pretrained_weight=None):
        super(LSTMBiA, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // 2 if bi_direction else hidden_dim
        self.vocab_num = vocab_num
        self.batch_size = batch_size
        self.conv_num_layer = num_layer
        self.model_num_layer = model_num_layer
        self.bi_direction = bi_direction
        self.word_embedding = nn.Embedding(vocab_num, embedding_dim, padding_idx=0)
        if pretrained_weight is not None:
            self.word_embedding.load_state_dict({'weight': pretrained_weight})

        self.sent_lstm = SentLSTM(embedding_dim, self.hidden_dim, batch_size)
        self.conv_lstm = nn.LSTM(hidden_dim + 3, self.hidden_dim, dropout=0 if self.conv_num_layer == 1 else dropout,
                                 num_layers=self.conv_num_layer, bidirectional=self.bi_direction)
        self.conv_hidden = self.init_hidden(self.batch_size, self.conv_num_layer)

        self.similarity = nn.Linear(hidden_dim * 3, 1)

        self.model_lstm = nn.LSTM(hidden_dim, self.hidden_dim, dropout=dropout, num_layers=self.model_num_layer,
                                  bidirectional=self.bi_direction)
        self.model_hidden = self.init_hidden(self.batch_size, self.model_num_layer)
        self.layer1 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, 1)  # Giving the final prediction
        self.final = nn.Sigmoid()

    def init_hidden(self, batch_size, num_layer):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi * num_layer, batch_size, self.hidden_dim).cuda(),
                    torch.randn(bi * num_layer, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(bi * num_layer, batch_size, self.hidden_dim),
                    torch.randn(bi * num_layer, batch_size, self.hidden_dim))

    def forward(self, target_conv, user_history):
        self.conv_hidden = self.init_hidden(len(target_conv), self.conv_num_layer)
        self.model_hidden = self.init_hidden(len(target_conv), self.model_num_layer)

        conv_reps = []
        conv_turn_nums = []
        for conv in target_conv:
            turn_num = 0
            sent_lens = []
            for turn in conv:
                if turn[3] == 0:
                    break
                turn_num += 1
                zero_num = torch.sum(turn[3:] == 0)  # find if there are 0s for padding
                sent_lens.append(len(turn) - 3 - zero_num)
            turn_infos = conv[:turn_num, :3].float()
            if torch.cuda.is_available():  # run in GPU
                turn_infos = turn_infos.cuda()
                sent_reps = self.sent_lstm(self.word_embedding(conv[:turn_num, 3:].cuda()), torch.LongTensor(sent_lens).cuda())
            else:
                sent_reps = self.sent_lstm(self.word_embedding(conv[:turn_num, 3:]), torch.LongTensor(sent_lens))
            conv_reps.append(torch.cat([sent_reps, turn_infos], dim=1))
            conv_turn_nums.append(turn_num)
        sorted_conv_turn_nums, sorted_conv_indices = torch.sort(torch.LongTensor(conv_turn_nums), descending=True)
        _, desorted_conv_indices = torch.sort(sorted_conv_indices, descending=False)
        sorted_conv_reps = []
        for index in sorted_conv_indices:
            sorted_conv_reps.append(conv_reps[index])
        paded_convs = rnn_utils.pad_sequence(sorted_conv_reps)
        packed_convs = rnn_utils.pack_padded_sequence(paded_convs, sorted_conv_turn_nums)
        conv_out, self.conv_hidden = self.conv_lstm(packed_convs, self.conv_hidden)
        conv_out = rnn_utils.pad_packed_sequence(conv_out, batch_first=True)[0]
        conv_out = conv_out[desorted_conv_indices]

        his_lens = []
        for one_his in user_history:
            his_len = 0
            for sent in one_his:
                if sent[0] == 0:
                    break
                his_len += 1
            his_lens.append(his_len)
        sorted_his_lens, sorted_his_indices = torch.sort(torch.LongTensor(his_lens), descending=True)
        _, desorted_his_indices = torch.sort(sorted_his_indices, descending=False)
        sorted_user_history = user_history[sorted_his_indices]
        his_out = []
        his_num = 0
        for one_his in sorted_user_history:
            sent_lens = []
            for sent in one_his[:sorted_his_lens[his_num]]:
                zero_num = torch.sum(sent == 0)  # find if there are 0s for padding
                sent_lens.append(len(sent) - zero_num)
            if torch.cuda.is_available():  # run in GPU
                his_out.append(self.sent_lstm(self.word_embedding(one_his[:sorted_his_lens[his_num]].cuda()), torch.LongTensor(sent_lens).cuda()))
            else:
                his_out.append(self.sent_lstm(self.word_embedding(one_his[:sorted_his_lens[his_num]]), torch.LongTensor(sent_lens)))
            his_num += 1
        his_out = rnn_utils.pad_sequence(his_out, batch_first=True)
        his_out = his_out[desorted_his_indices]

        batch_size = conv_out.size(0)
        conv_sent_len = conv_out.size(1)
        his_sent_len = his_out.size(1)
        hidden_dim = conv_out.size(-1)

        conv_rep = conv_out.repeat(1, 1, his_sent_len).view(batch_size, conv_sent_len * his_sent_len, -1)
        his_rep = his_out.repeat(1, conv_sent_len, 1)
        sim_matrix = torch.cat([conv_rep, his_rep, conv_rep * his_rep], -1).view(batch_size, conv_sent_len, his_sent_len, -1)
        sim_matrix = self.similarity(sim_matrix)
        sim_matrix = sim_matrix.squeeze(-1)
        # print sim_matrix

        atten_c2h = F.softmax(sim_matrix, dim=-1)
        atten_c2h = atten_c2h.unsqueeze(-1).repeat(1, 1, 1, hidden_dim)
        atten_c2h = atten_c2h * his_out.unsqueeze(1)
        atten_c2h = atten_c2h.sum(2)
        # print F.softmax(sim_matrix, dim=-1)
        # print F.softmax(sim_matrix.max(2)[0], dim=-1)

        atten_h2c = F.softmax(sim_matrix.max(2)[0], dim=-1)
        atten_h2c = atten_h2c.unsqueeze(-1) * conv_out
        atten_h2c = atten_h2c.sum(1)
        atten_h2c = atten_h2c.unsqueeze(1).repeat(1, conv_sent_len, 1)

        conv_rep_atten = torch.cat([conv_out, atten_c2h, conv_out * atten_c2h, conv_out * atten_h2c], dim=-1)
        conv_rep_atten = F.relu(self.layer1(conv_rep_atten))
        conv_rep_atten = F.relu(self.layer2(conv_rep_atten))
        sorted_conv_rep_atten = conv_rep_atten[sorted_conv_indices]
        packed_conv_rep_atten = rnn_utils.pack_padded_sequence(sorted_conv_rep_atten, sorted_conv_turn_nums, batch_first=True)
        _, self.model_hidden = self.model_lstm(packed_conv_rep_atten, self.model_hidden)
        if self.bi_direction:
            model_out = torch.cat([self.model_hidden[0][-2], self.model_hidden[0][-1]], dim=1)
        else:
            model_out = self.model_hidden[0][-1]
        model_out = model_out[desorted_conv_indices]

        conv_labels = self.final(self.out_layer(model_out).view(-1))
        return conv_labels




