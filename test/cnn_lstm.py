import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import hyperparams
torch.manual_seed(hyperparams.seed_num)
random.seed(hyperparams.seed_num)

class  CLSTM(nn.Module):
    
    def __init__(self, args):
        super(CLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes
        self.embed = nn.Embedding(V, D)
        if args.word_Embedding:
            pretrained_weight = np.array(args.pretrained_weight)
            self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))

        # CNN
        KK = []
        for K in Ks:
            KK.append( K + 1 if K % 2 == 0 else K)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        self.convs1 = [nn.Conv2d(Ci, D, (K, D), stride=1, padding=(K//2, 0)) for K in KK]
        # LSTM
        self.lstm = nn.LSTM(D, self.hidden_dim, num_layers= self.num_layers, dropout=args.dropout)
        self.hidden = self.init_hidden(self.num_layers, args.batch_size)
        # linear
        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)
        # dropout
        self.dropout = nn.Dropout(args.dropout)


    def init_hidden(self,num_layers, batch_size):
        # the first is the hidden h
        # the second is the cell  c
        return (Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)),
                 Variable(torch.zeros(1 * num_layers, batch_size, self.hidden_dim)))


    def forward(self, x):
        embed = self.embed(x)
        # CNN
        cnn_x = embed
        cnn_x = self.dropout(cnn_x)
        cnn_x = cnn_x.unsqueeze(1)
        cnn_x = [F.relu(conv(cnn_x)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
        cnn_x = torch.cat(cnn_x, 0)
        cnn_x = torch.transpose(cnn_x, 1, 2)
        # LSTM
        lstm_out, self.hidden = self.lstm(cnn_x, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        # linear
        cnn_lstm_out = self.hidden2label1(F.tanh(lstm_out))
        cnn_lstm_out = self.hidden2label2(F.tanh(cnn_lstm_out))
        # output
        logit = cnn_lstm_out

        return logit
