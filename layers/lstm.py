import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LMLSTM(nn.Module):

    def __init__(self, config):
        super(LMLSTM, self).__init__()

        # parsing config
        self.vocab_size = config['vocab_size']
        self.embed_dim = config.get('embed_dim', 125)
        self.lstm_dim = config.get('lstm_dim', 256)
        self.num_layers = config.get('num_layers', 1)

        # some layers
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.do = nn.Dropout(0.2)
        self.rnn = nn.LSTM(self.embed_dim, self.lstm_dim, 
                num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(self.lstm_dim, self.vocab_size)


    def forward(self, X, state):
        # first embed it
        embed = self.embed(X)
        out, state = self.rnn(embed)  # (batch_size, max_seq, lstm_dim)
        out = self.out(out)  # (batch_size, max_seq, vocab_size)
        return out, state

    def init_state(self, batch_size):
        return (
                torch.zeros(self.num_layers, batch_size, self.lstm_dim),
                torch.zeros(self.num_layers, batch_size, self.lstm_dim)
            )

