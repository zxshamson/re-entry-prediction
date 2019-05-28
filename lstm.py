import torch
import torch.nn.functional as F
from torch import nn, optim
import torch.nn.utils.rnn as rnn_utils


class SentLSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_num, sent_hidden_dim, batch_size, bi_direction, pretrained_weight):
        super(SentLSTM, self).__init__()
        self.sent_hidden_dim = sent_hidden_dim // 2 if bi_direction else sent_hidden_dim
        self.bi_direction = bi_direction
        self.word_embedding = nn.Embedding(vocab_num, embedding_dim, padding_idx=0)
        if pretrained_weight is not None:
            self.word_embedding.load_state_dict({'weight': pretrained_weight})
        self.sent_lstm = nn.LSTM(embedding_dim, self.sent_hidden_dim, bidirectional=bi_direction)
        self.sent_hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi, batch_size, self.sent_hidden_dim).cuda(),
                    torch.randn(bi, batch_size, self.sent_hidden_dim).cuda())
        else:
            return (torch.randn(bi, batch_size, self.sent_hidden_dim),
                    torch.randn(bi, batch_size, self.sent_hidden_dim))

    def forward(self, sentences, sent_lens):
        self.sent_hidden = self.init_hidden(len(sentences))
        turn_infos = sentences[:, :3].float()
        sorted_sent_lens, indices = torch.sort(sent_lens, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        sorted_sentences = sentences[indices]
        embeds = self.word_embedding(sorted_sentences[:, 3:])
        packed_embeds = rnn_utils.pack_padded_sequence(embeds, sorted_sent_lens, batch_first=True)
        lstm_out, self.sent_hidden = self.sent_lstm(packed_embeds, self.sent_hidden)
        if self.bi_direction:
            sent_reps = torch.cat([self.sent_hidden[0][-2], self.sent_hidden[0][-1]], dim=1)
            sent_reps = sent_reps[desorted_indices]
        else:
            sent_reps = self.sent_hidden[0][-1][desorted_indices]
        sent_reps = torch.cat([sent_reps, turn_infos], dim=1)
        """
        embeds = self.word_embedding(sentences[:, 3:])
        lstm_out, self.sent_hidden = self.sent_lstm(embeds, self.sent_hidden)
        sent_reps = lstm_out[0][sent_lens[0]-1].view(1, -1)
        for sent in range(1, len(sentences)):
            sent_reps = torch.cat([sent_reps, lstm_out[sent][sent_lens[sent]-1].view(1, -1)], dim=0)
        sent_reps = torch.cat([sent_reps, turn_infos], dim=1)
        """
        return sent_reps


class LSTM(nn.Module):
    def __init__(self, embedding_dim, vocab_num, hidden_dim, batch_size, dropout, num_layer, bi_direction, pretrained_weight=None):
        super(LSTM, self).__init__()
        self.conv_hidden_dim = hidden_dim // 2 if bi_direction else hidden_dim
        self.batch_size = batch_size
        self.bi_direction = bi_direction
        self.num_layer = num_layer
        self.sent_lstm = SentLSTM(embedding_dim, vocab_num, hidden_dim, batch_size, bi_direction, pretrained_weight)
        self.conv_lstm = nn.LSTM(hidden_dim+3, self.conv_hidden_dim, dropout=0 if self.num_layer == 1 else dropout,
                                 num_layers=num_layer, bidirectional=bi_direction)
        self.conv_hidden = self.init_hidden(self.batch_size)
        self.hidden2label = nn.Linear(hidden_dim, 1)
        self.final = nn.Sigmoid()

    def init_hidden(self, batch_size):
        bi = 2 if self.bi_direction else 1
        if torch.cuda.is_available():  # run in GPU
            return (torch.randn(bi * self.num_layer, batch_size, self.conv_hidden_dim).cuda(),
                    torch.randn(bi * self.num_layer, batch_size, self.conv_hidden_dim).cuda())
        else:
            return (torch.randn(bi * self.num_layer, batch_size, self.conv_hidden_dim),
                    torch.randn(bi * self.num_layer, batch_size, self.conv_hidden_dim))

    def forward(self, convs):
        self.conv_hidden = self.init_hidden(len(convs))
        conv_reps = []
        conv_turn_nums = []
        for conv in convs:
            turn_num = 0
            sent_lens = []
            for turn in conv:
                if turn[3] == 0:
                    break
                turn_num += 1
                zero_num = torch.sum(turn[3:] == 0)    # find if there are 0s for padding
                sent_lens.append(len(turn) - 3 - zero_num)
            if torch.cuda.is_available():  # run in GPU
                sent_reps = self.sent_lstm(conv[:turn_num].cuda(), torch.LongTensor(sent_lens).cuda())
            else:
                sent_reps = self.sent_lstm(conv[:turn_num], torch.LongTensor(sent_lens))
            conv_reps.append(sent_reps)
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
            true_out = torch.cat([self.conv_hidden[0][-2], self.conv_hidden[0][-1]], dim=1)
            true_out = true_out[desorted_indices]
        else:
            true_out = self.conv_hidden[0][-1][desorted_indices]
        conv_labels = self.final(self.hidden2label(true_out).view(-1))
        return conv_labels


