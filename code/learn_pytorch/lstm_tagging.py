#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : lstm_tagging.py
# @Author: harry
# @Date  : 18-8-2 上午11:07
# @Desc  : Yet another lstm in pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_sequence(seq, word_to_ix):
    idxs = [word_to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for s, tags in training_data:
    for word in s:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, emdedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, emdedding_dim)
        self.lstm = nn.LSTM(emdedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),  # h0
                torch.zeros(1, 1, self.hidden_dim))  # c0

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1),
            self.hidden
        )
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print('Before training')
    print(tag_scores)

for epoch in range(300):
    print('epoch {}'.format(epoch + 1))
    running_loss = 0.0
    for s, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        s_in = prepare_sequence(s, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        tag_scores = model(s_in)
        loss = loss_function(tag_scores, targets)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    print('loss: {:.6f}'.format(running_loss / len(training_data)))

with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print('After training')
    print(tag_scores)
