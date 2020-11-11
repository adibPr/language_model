#!/usr/bin/env python

import os
import re
import random

import torch
from torch.utils.data import Dataset, DataLoader

path_this = os.path.abspath (os.path.dirname (__file__))
from vocab import Vocab


def load_meditation(
            path=os.path.join (path_this, 'data', 'meditations.mb.txt'),
            split=(0.7, 0.15, 0.15), # training, validation, testing
            shuffle=True
        ):

    with open (path) as f_:
        lines = f_.read ().splitlines ()
        
    # removing header and footer
    lines = lines[11:4317]

    # removing empty line or line with only ------ and line starts with BOOK
    separator = "-"*70
    lines = [l for l in lines if l.strip ()\
                    and l.strip() != separator\
                    and not l.strip().startswith ('BOOK')
                ]

    # add space between non characters
    lines = [re.sub (r'([\',!\.\?:;/\\-])', r' \1', l.lower(), flags=re.I) for l in lines]

    random.shuffle (lines)
    idx_train = (0, int (split[0] * len (lines)))
    idx_val = (idx_train[-1], idx_train[-1] + int (split[1] * len (lines)))
    idx_test = (idx_val[-1], -1)
    
    return (
            lines[:idx_train[1]], 
            lines[idx_val[0]:idx_val[1]], 
            lines[idx_test[0]:]
        )

def pad_sequences(
            sequences, 
            maxlen=None, 
            padding='pre', 
            truncating='pre',
            pad_token=0
        ):

    sequences_padded = []
    if maxlen is None:
        maxlen = max (map (len, sequences))

    for seq in sequences:
        if truncating == 'pre':
            seq = seq[-maxlen:]
        else:
            seq = seq[:maxlen]

        tot_pad = max (maxlen - len (seq), 0)
        if padding == 'pre':
            seq = [pad_token] * tot_pad + seq
        else:
            seq = seq + ([pad_token] * tot_pad)

        sequences_padded.append (seq)

    return sequences_padded


class LangModelDataset(Dataset):

    def __init__ (self, data, vocab=None, **kwargs):
        if not vocab:
            self.vocab = Vocab (
                    num_words=kwargs.get ("num_words", None), 
                    is_lower=kwargs.get ("is_lower", True)
                )
            self.vocab.fit (data)
        else:
            # assuming no need another fit
            self.vocab = vocab

        self.data = self.vocab.transform (data)
        self.data_pad = pad_sequences (
                self.data, 
                maxlen=kwargs.get('maxlen', None),
                padding=kwargs.get("padding", "pre"),
                truncating=kwargs.get("truncating", "pre"),
                pad_token=self.vocab.stoi["_pad_"]
            )

    def __len__(self):
        return len (self.data)

    def __getitem__(self, idx):
        # return tuple, seq training and its target
        return (
                torch.tensor(self.data_pad[idx][:-1], dtype=torch.long), 
                torch.tensor(self.data_pad[idx][1:], dtype=torch.long)
            )


if __name__ == "__main__":
    train_data, val_data, test_data = load_meditation ()
    vocab = Vocab (num_words=10000)
    vocab.fit (train_data)
    LM = LangModelDataset (train_data, vocab, maxlen=20)
