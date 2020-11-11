#!/usr/bin/env python

import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from vocab import Vocab
from data import load_meditation, LangModelDataset
from layers.lstm import LMLSTM

# config
config = {
        # vocab parameter
        "vocab_size":10000,
        "seq_size": 20,

        # model parameter
        "embed_dim": 125,
        "lstm_dim": 125,
        "num_layers": 1,

        # training parameter
        "batch_size": 100,
        "device": "gpu" if torch.cuda.is_available() else "cpu",
        "epoch": 5,
        "learning_rate": 0.01
}

# building vocab
train_data, val_data, test_data = load_meditation ()
vocab = Vocab (num_words=config['vocab_size'])
vocab.fit (train_data)

# building dataset
train_dataset = LangModelDataset(train_data, vocab, maxlen=config['vocab_size'])
val_dataset = LangModelDataset(val_data, vocab, maxlen=config['vocab_size'])
test_dataset = LangModelDataset(test_data, vocab, maxlen=config['vocab_size'])

# building iterator
train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'])
val_iterator = DataLoader(val_dataset, batch_size=config['batch_size'])
test_iterator = DataLoader(test_dataset, batch_size=config['batch_size'])

# initialize model, optim and losses
model = LMLSTM(config)
optim = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
criterion = nn.NLLLoss()
device = torch.device(config["device"])
model.to(device)

# training
for epoch in range(config["epoch"]):
    print("Epoch --- {}".format(epoch+1))
    step_loss = 0

    model.train()
    for step, (seq_in, seq_label) in enumerate(train_iterator):
        # send to device
        seq_in = seq_in.to(device)
        seq_label = seq_label.to(device)

        # clear all gradient
        optim.zero_grad()
        
        # forward pass
        out = model(seq_in)

        # compute loss
        loss = criterion(out, seq_label)
        # compute gradient 
        loss.backward()

        # update weight
        optim.step()

        step_loss += loss.item()

        if step%100 == 99:
            print("Epoch {}, step {}, avg loss {:.2F}".format(
                epoch+1, step+1, step_loss/100
            ))
            step_loss = 0
print("Done")
