#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : CBOW.py
# @Author: harry
# @Date  : 18-8-1 下午10:52
# @Desc  : Yet another CBOW Model written in pytorch
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
MODEL_FILE = './model/cbow.pkl'


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(2 * context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = F.log_softmax(self.linear2(out))
        return out


class CBOWWrapper():
    def __init__(self, vocab, word_to_ix, ix_to_word,
                 embedding_dim, context_size, learning_rate):
        # params
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        # CBOW model
        self.net = CBOW(self.vocab_size, self.embedding_dim, self.context_size)

        # Try to train on GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Device: ", self.device)
        self.net.to(self.device)

        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)

    @staticmethod
    def make_context_vector(context, word_to_ix):
        idxs = [word_to_ix[w] for w in context]
        return torch.LongTensor(idxs)

    @staticmethod
    def make_target_vector(target, word_to_ix):
        idxs = [word_to_ix[target]]
        return torch.LongTensor(idxs)

    def load_model(self):
        if os.path.isfile(MODEL_FILE):
            self.net.load_state_dict(torch.load(MODEL_FILE))
            print('Model loaded from {}'.format(MODEL_FILE))
            return True
        else:
            print('No model file found'.format(MODEL_FILE))
            return False

    def save_model(self):
        torch.save(self.net.state_dict(), MODEL_FILE)
        print('Model saved in {}'.format(MODEL_FILE))

    def train(self, epochs, data):
        for epoch in range(epochs):
            print('epoch {}'.format(epoch + 1))
            running_loss = 0.0
            for d in data:
                context, target = d
                context = self.make_context_vector(context, self.word_to_ix).to(self.device)
                target = self.make_target_vector(target, self.word_to_ix).to(self.device)
                self.optimizer.zero_grad()

                # forward propagation
                out = self.net(context)
                loss = self.criterion(out, target)
                running_loss += loss.item()

                # backward propagation
                loss.backward()
                self.optimizer.step()
            print('loss: {:.6f}'.format(running_loss / len(data)))
        print('Finished training')
        self.save_model()

    def get_embedding(self, word):
        with torch.no_grad():
            embeds = self.net.embeddings
            ix = torch.LongTensor([self.word_to_ix[word]]).to(self.device)
            return embeds(ix)

    @staticmethod
    def cosine_sim(v1, v2):
        return np.dot(v1, np.transpose(v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def test(self):
        word1 = 'are'
        word2 = 'is'
        word3 = 'Computational'
        embeds1 = self.get_embedding(word1).cpu()
        embeds2 = self.get_embedding(word2).cpu()
        embeds3 = self.get_embedding(word3).cpu()
        v1 = embeds1.numpy()
        v2 = embeds2.numpy()
        v3 = embeds3.numpy()
        print('cosine sim between {} and {} is {}'.format(word1, word2, self.cosine_sim(v1, v2)))
        print('cosine sim between {} and {} is {}'.format(word1, word3, self.cosine_sim(v1, v3)))


def main():
    # print(data[0])
    # print(make_context_vector(data[0][0], word_to_ix))

    # prepare vocab
    vocab = set(raw_text)
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # prepare data
    data = []
    for i in range(2, len(raw_text) - 2):
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))
    # params
    embedding_dim = 128
    learning_rate = 0.001
    train_epochs = 200

    cbow = CBOWWrapper(vocab, word_to_ix, ix_to_word,
                       embedding_dim, CONTEXT_SIZE, learning_rate)

    if not cbow.load_model():
        cbow.train(train_epochs, data)

    cbow.test()


if __name__ == '__main__':
    main()
