import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import random
import configs.config_sse as config


class SSENet(nn.Module):

    def __init__(self, input_dim):
        super(SSENet, self).__init__()
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers

        V = len(config.IUPAC_VOCAB)
        self.embed = nn.Embedding(V, config.embed_dim, padding_idx=0)

        # CNN
        # self.convs1 = [nn.Conv2d(1, config.kernel_num, (K, input_dim), padding=(K // 2, 0), stride=1) for K in
        #                config.kernel_sizes]
        # for cnn cuda
        # for conv in self.convs1:
        #     conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=config.dropout,
                              bidirectional=True, bias=True)

        # linear
        self.cnn_lstm_len = config.lstm_hidden_dim * 2  # + config.kernel_num * 3
        self.hidden2label1 = nn.Linear(self.cnn_lstm_len, self.cnn_lstm_len // 2)
        self.hidden2label2 = nn.Linear(self.cnn_lstm_len // 2, config.num_labels)

        # dropout
        self.dropout = nn.Dropout(config.dropout)

    def init_weights(self):
        init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))
        init.xavier_normal_(self.bilstm.all_weights[0][1], gain=np.sqrt(2))
        init.xavier_uniform_(self.hidden2label1.weight, gain=1)
        init.xavier_uniform_(self.hidden2label2.weight, gain=1)
        init.xavier_uniform_(self.embed.weight.data, gain=1)
        # for conv in self.convs1:
        #     init.xavier_uniform_(conv.weight.data, gain=1)

    def forward(self, sequence, profile):
        # if real_profile is not None:
        #     profile = torch.cat([profile, real_profile], dim=-1)

        _sequence = torch.transpose(sequence, 0, 1)
        _profile = torch.transpose(profile, 0, 1)

        embed = self.embed(_sequence)  # 44 x 16 x 300

        input = torch.cat([embed, _profile], dim=2)  # [44 x 16 x 321]

        # BiLSTM
        bilstm_out, _ = self.bilstm(input)  # 44 x 16 x 600
        bilstm_out = torch.transpose(bilstm_out, 0, 1)  # 16 x 44 x 600

        # CNN and BiLSTM CAT
        # cnn_lstm = cnn_lstm.contiguous().view(-1, self.cnn_lstm_len)  # 704 x 1200

        # linear
        _x = self.hidden2label1(bilstm_out)

        x = self.hidden2label2(_x)
        # out = x.view(profile.shape[0], -1, config.num_labels)  # 16 x 44 x 3
        out = x

        return out, bilstm_out

