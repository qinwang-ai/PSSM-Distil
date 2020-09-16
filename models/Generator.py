import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random
import configs.config_gen as config
import configs.config_sse as config_sse

"""
    Neural Network: CNN_BiLSTM
    Detail: the input crosss cnn model and LSTM model independly, then the result of both concat
"""

class Generator(nn.Module):

    def __init__(self, pure_bert=False):
        super(Generator, self).__init__()
        self.hidden_dim = config.lstm_hidden_dim
        self.num_layers = config.lstm_num_layers
        self.is_softmax = True
        V = len(config_sse.IUPAC_VOCAB)
        if pure_bert:
            input_dim = config.embed_dim + config_sse.profile_width
        else:
            input_dim = config.embed_dim + config_sse.profile_width * 2

        self.embed = nn.Embedding(V, config.embed_dim, padding_idx=0)
        # pretrained  embedding

        # CNN
        self.convs1 = [nn.Conv2d(1, config.kernel_num, (K, input_dim), padding=(K // 2, 0), stride=1) for K in
                       config.kernel_sizes]
        # for cnn cuda
        for conv in self.convs1:
            conv.cuda()

        # BiLSTM
        self.bilstm = nn.LSTM(input_dim, self.hidden_dim, num_layers=self.num_layers, dropout=config.dropout,
                              bidirectional=True, bias=True)
        # linear
        self.cnn_lstm_len = config.lstm_hidden_dim * 2 + config.kernel_num * 3
        self.hidden2label1 = nn.Linear(self.cnn_lstm_len, self.cnn_lstm_len // 2)
        self.hidden2label2 = nn.Linear(self.cnn_lstm_len // 2, config_sse.profile_width)

        # dropout
        # self.dropout = nn.Dropout(config.dropout)

    def init_weights(self):
        init.xavier_normal_(self.bilstm.all_weights[0][0], gain=np.sqrt(2))
        init.xavier_normal_(self.bilstm.all_weights[0][1], gain=np.sqrt(2))
        init.xavier_uniform_(self.hidden2label1.weight, gain=1)
        init.xavier_uniform_(self.hidden2label2.weight, gain=1)
        init.xavier_uniform_(self.embed.weight.data, gain=1)
        for conv in self.convs1:
            init.xavier_uniform_(conv.weight.data, gain=1)

    def forward(self, sequence, profile, softmax=False):
        # sequence 16 x 44
        # profile 16 x 44 x 21
        self.is_softmax = softmax
        _sequence = torch.transpose(sequence, 0, 1) # 44 x 16
        _profile = torch.transpose(profile, 0, 1) # 44 x 16

        embed = self.embed(_sequence)  # 44 x 16 x 300
        input = torch.cat([embed, _profile], dim=2)  # [44 x 16 x 321]

        # CNN
        cnn_x = torch.transpose(input, 0, 1)  # [16 x 44 x 790]
        cnn_x = cnn_x.unsqueeze(1)  # [16 x 1 x 44 x 790]
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [16 x 200 x 44]
        cnn_x = torch.cat(cnn_x, dim=1)  # 16 x 600 x 44

        # BiLSTM
        bilstm_out, _ = self.bilstm(input)  # 44 x 16 x 600
        bilstm_out = torch.transpose(bilstm_out, 0, 1)  # 16 x 44 x 600
        bilstm_out = torch.transpose(bilstm_out, 1, 2)  # 16 x 600 x 44

        # CNN and BiLSTM CAT
        cnn_lstm = torch.cat([cnn_x, bilstm_out], dim=1)  # 16 x 1200 x 44
        cnn_lstm = torch.transpose(cnn_lstm, 1, 2)  # 16 x 44 x 1200
        # cnn_lstm = cnn_lstm.contiguous().view(-1, self.cnn_lstm_len)  # 704 x 1200

        # linear
        x = self.hidden2label1(cnn_lstm)
        x = self.hidden2label2(x)

        # out = x.view(sequence.shape[0], -1, config_sse.profile_width)  # 16 x 44 x 21

        # if softmax:
        #     out = F.softmax(out, dim=1)
        out = x
        return out

