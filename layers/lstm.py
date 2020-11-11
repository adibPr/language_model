import numpy as np
from torch import nn
from torch.nn import functional as F


class LMLSTM(nn.Module):

    def __init__(self, config):
        super(LMLSTM, self).__init__()

        # parsing config
        self.config = config
        vocab_size = config['vocab_size']
        embed_dim = config.get('embed_dim', 125)
        lstm_dim = config.get('lstm_dim', 256)
        num_layers=config.get('num_layers', 1)

        # some layers
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.do = nn.Dropout(0.2)
        self.rnn = nn.LSTM(embed_dim, lstm_dim, 
                num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(lstm_dim, vocab_size)


    def forward(self, X):
        # first embed it
        embed = self.embed(X)
        out, state = self.do(self.rnn(embed))  # (batch_size, max_seq, lstm_dim)
        out = self.out(out)  # (batch_size, max_seq, vocab_size)
        return F.log_softmax(out)

