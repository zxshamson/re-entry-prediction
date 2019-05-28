import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class SentLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, batch_size, bi_direction):
        super(SentLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.bi_direction = bi_direction
        self.sent_lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bi_direction)
        self.sent_hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
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


class LSTMMEM(nn.Module):
    def __init__(self, embedding_dim, vocab_num, hidden_dim, hop_num, batch_size, dropout, num_layer, bi_direction=True, pretrained_weight=None):
        super(LSTMMEM, self).__init__()
        self.hop_num = hop_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // 2 if bi_direction else hidden_dim
        self.vocab_num = vocab_num
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.bi_direction = bi_direction
        self.word_embedding = nn.Embedding(vocab_num, embedding_dim, padding_idx=0)
        self.A = nn.ModuleList([SentLSTM(embedding_dim, self.hidden_dim, batch_size, bi_direction) for _ in range(hop_num+1)])
        if pretrained_weight is None:
            nn.init.xavier_normal_(self.word_embedding.weight)
        else:
            self.word_embedding.load_state_dict({'weight': pretrained_weight})
        self.B = self.A[0]
        self.conv_lstm = nn.LSTM(hidden_dim + 3, self.hidden_dim, dropout=0 if self.num_layer == 1 else dropout,
                                 num_layers=num_layer, bidirectional=self.bi_direction)
        self.conv_hidden = self.init_hidden(self.batch_size)
        self.embed2label = nn.Linear(hidden_dim, 1)  # Giving the final prediction
        # nn.init.xavier_normal_(self.embed2label.weight)
        self.final = nn.Sigmoid()

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi * self.num_layer, batch_size, self.hidden_dim).cuda(),
                    torch.randn(bi * self.num_layer, batch_size, self.hidden_dim).cuda())
        else:
            return (torch.randn(bi * self.num_layer, batch_size, self.hidden_dim),
                    torch.randn(bi * self.num_layer, batch_size, self.hidden_dim))

    def forward(self, target_conv, user_history):
        self.conv_hidden = self.init_hidden(len(target_conv))
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
                sent_reps = self.B(self.word_embedding(conv[:turn_num, 3:].cuda()), torch.LongTensor(sent_lens).cuda())
            else:
                sent_reps = self.B(self.word_embedding(conv[:turn_num, 3:]), torch.LongTensor(sent_lens))
            conv_reps.append(torch.cat([sent_reps, turn_infos], dim=1))
            conv_turn_nums.append(turn_num)
        sorted_conv_turn_nums, indices = torch.sort(torch.LongTensor(conv_turn_nums), descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_conv_reps = []
        for index in indices:
            sorted_conv_reps.append(conv_reps[index])
        paded_convs = rnn_utils.pad_sequence(sorted_conv_reps)
        packed_convs = rnn_utils.pack_padded_sequence(paded_convs, sorted_conv_turn_nums)
        lstm_out, self.conv_hidden = self.conv_lstm(packed_convs, self.conv_hidden)
        if self.bi_direction:
            u = torch.cat([self.conv_hidden[0][-2], self.conv_hidden[0][-1]], dim=1)
            u = u[desorted_indices]
        else:
            u = self.conv_hidden[0][-1][desorted_indices]

        max_his_len = user_history.size(1)
        embed_dim = u.size(-1)
        for hop in range(self.hop_num):
            m_A = []
            m_C = []
            for one_his in user_history:
                sent_lens = []
                his_len = 0
                for sent in one_his:
                    if sent[0] == 0:
                        break
                    his_len += 1
                    zero_num = torch.sum(sent == 0)  # find if there are 0s for padding
                    sent_lens.append(len(sent) - zero_num)
                if torch.cuda.is_available():  # run in GPU
                    m_a = self.A[hop](self.word_embedding(one_his[: his_len].cuda()), torch.LongTensor(sent_lens).cuda())
                    m_c = self.A[hop+1](self.word_embedding(one_his[: his_len].cuda()), torch.LongTensor(sent_lens).cuda())
                    m_A.append(torch.cat([m_a, torch.Tensor([[0] * embed_dim] * (max_his_len - his_len)).cuda()], dim=0).unsqueeze(0))
                    m_C.append(torch.cat([m_c, torch.Tensor([[0] * embed_dim] * (max_his_len - his_len)).cuda()], dim=0).unsqueeze(0))
                else:
                    m_a = self.A[hop](self.word_embedding(one_his[: his_len]), torch.LongTensor(sent_lens))
                    m_c = self.A[hop+1](self.word_embedding(one_his[: his_len]), torch.LongTensor(sent_lens))
                    m_A.append(torch.cat([m_a, torch.Tensor([[0] * embed_dim] * (max_his_len - his_len))], dim=0).unsqueeze(0))
                    m_C.append(torch.cat([m_c, torch.Tensor([[0] * embed_dim] * (max_his_len - his_len))], dim=0).unsqueeze(0))
            m_A = torch.cat(m_A, dim=0)
            m_C = torch.cat(m_C, dim=0)
            # print m_A.size()

            prob = torch.bmm(m_A, u.unsqueeze(2)).squeeze(-1)
            # m_A: (batch_size, his_num, embed_dim), u.unsqueeze(2): (batch_size, embed_dim, 1)
            prob = F.softmax(prob, -1).unsqueeze(1)
            # prob: (batch_size, 1, his_num)
            o = torch.bmm(prob, m_C).squeeze(1)
            # o: (batch_size, embed_dim)
            u = o + u

        labels = self.final(self.embed2label(u).view(-1))
        return labels








