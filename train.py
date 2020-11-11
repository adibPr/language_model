#!/usr/bin/env python

import os
import sys

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from vocab import Vocab
from data import load_meditation, load_penntrebank, LangModelDataset
from layers.lstm import LMLSTM

# config
config = {
        # vocab parameter
        "vocab_size":10000,
        "seq_size": 30,

        # model parameter
        "embed_dim": 128,
        "lstm_dim": 1025,
        "num_layers": 1,

        # training parameter
        "batch_size": 20,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "epoch": 5,
        "learning_rate": 0.02
}

# building vocab
# train_data, val_data, test_data = load_meditation (maxlen=config['seq_size'])
train_data, val_data, test_data = load_penntrebank ()
vocab = Vocab (num_words=config['vocab_size'])
vocab.fit (train_data)

# building dataset
train_dataset = LangModelDataset(train_data, vocab, maxlen=config['seq_size'])
val_dataset = LangModelDataset(val_data, vocab, maxlen=config['seq_size'])
test_dataset = LangModelDataset(test_data, vocab, maxlen=config['seq_size'])

# building iterator
train_iterator = DataLoader(train_dataset, batch_size=config['batch_size'], drop_last=True)
val_iterator = DataLoader(val_dataset, batch_size=config['batch_size'])
test_iterator = DataLoader(test_dataset, batch_size=config['batch_size'])

# initialize model, optim and losses
model = LMLSTM(config)
optim = optim.Adam(model.parameters(), lr=config["learning_rate"])
criterion = nn.CrossEntropyLoss()
device = torch.device(config["device"])
model.to(device)

# training
for epoch in range(config["epoch"]):
    print("Epoch --- {}".format(epoch+1))
    step_loss = []

    model.train()
    state = tuple(map (lambda s:s.to(device), model.init_state(config['batch_size'])))

    for step, (seq_in, seq_label) in enumerate(train_iterator):
        # send to device
        seq_in = seq_in.to(device)
        seq_label = seq_label.to(device)

        # clear all gradient
        optim.zero_grad()
        
        # forward pass
        out, state = model(seq_in, state)
        state = [s.detach() for s in state]

        # compute loss
        # reshaping output, NLLLoss requires (N, C, d) see: 
        # https://stackoverflow.com/questions/60121107/pytorch-nllloss-function-target-shape-mismatch
        loss = criterion(out.transpose(1,2), seq_label)
        # compute gradient 
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        # update weight
        optim.step()

        step_loss.append(loss.item())

        if step%100 == 0:
            print("Epoch {}, step {}, avg loss {:.4F}".format(
                epoch+1, step, sum(step_loss)/len(step_loss)
            ))
            step_loss = []
print("Done")

# testing
start_idx = 312
seq = [start_idx]# just random
seq = torch.tensor(seq, dtype=torch.long).unsqueeze(1) # batch_size, max_seq
word_seq = [vocab.itos[start_idx]]
max_word = 20

with torch.no_grad():
    model.eval()
    state = tuple(map (lambda s:s.to(device), model.init_state(1)))
    for i in range(max_word-1):
        seq = seq.to(device)

        out, state = model(seq, state)
        # out in log, make it prob, but take the last one first
        prob = out[0,-1,:].exp()

        # sample a word, greedy
        # must batch_size x 1, argmax return Long 
        # next_pred = torch.argmax(prob).unsqueeze(0).unsqueeze(0)

        # this one take label using multinomial, taken from
        # https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/language_model/main.py
        next_pred = torch.multinomial(prob, num_samples=1)
        next_pred = next_pred.unsqueeze(0)

        seq = torch.cat((seq, next_pred), dim=1)
        word_seq.append(vocab.itos[next_pred.item()])
print(" ".join (word_seq))
        
