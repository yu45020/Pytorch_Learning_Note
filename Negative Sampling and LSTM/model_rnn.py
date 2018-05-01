import logging
from itertools import chain
from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


# +++++++++++++++++++++++++++++++++++++
#           Prepare Dataloader      
# -------------------------------------

class PoetryRNNData(Dataset):
    def __init__(self, data, max_length=120):
        # when chunk size = 120, evenly divide; = 259, leave one out
        # most poetries have length around 40 - 80
        # data is nested list of word idx
        assert any(isinstance(i, list) for i in data)

        self.lengths = torch.LongTensor([min(len(x), max_length) - 1 for x in data])
        self.lens = len(self.lengths)
        # pad data
        max_len = min(max(self.lengths), max_length)
        self.data = torch.zeros((self.lens, max_len,)).long()
        self.target = torch.zeros((self.lens, max_len)).long()
        for i in range(self.lens):
            L = min(self.lengths[i], max_length)
            self.data[i, :L] = torch.LongTensor(data[i][:L])
            self.target[i, :L] = torch.LongTensor(data[i][1:(L + 1)])
        if use_cuda:
            self.data = self.data.cuda()
            self.target = self.target.cuda()
            self.lengths = self.lengths.cuda()

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        return self.data[index, :], self.target[index, :], self.lengths[index]


def prepare_data_loader(data, max_length, batch_size, shuffle):
    dataset = PoetryRNNData(data, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def sort_batches(batches):
    x, y, lengths = batches
    sorted_lens, sorted_idx = lengths.sort(dim=0, descending=True)

    sorted_x = x.index_select(dim=0, index=sorted_idx.squeeze())  # x[sorted_idx, :]
    sorted_y = y.index_select(dim=0, index=sorted_idx.squeeze())  # y[sorted_idx, :]

    pad_len = sorted_lens.squeeze()
    unpad_y = [sorted_y[i, :pad_len[i]] for i in range(len(pad_len))]
    unpad_y = torch.cat(unpad_y)

    sorted_x = sorted_x.contiguous()
    unpad_y = unpad_y.contiguous()
    #if use_cuda:
    #    sorted_x = sorted_x.cuda()
    #    unpad_y = unpad_y.cuda()

    return sorted_x, unpad_y, sorted_lens.squeeze()


# +++++++++++++++++++++++++++++++++++++
#           Prepare RNN model      
# -------------------------------------

class PoetryRNN(nn.Module):
    def __init__(self, pretrain_embed_layer,
                 rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                 dense_dim, dense_h_dropout,
                 freeze_embed=False):
        super(PoetryRNN, self).__init__()

        # pre-trained word embedding
        self.embed = nn.Embedding.from_pretrained(pretrain_embed_layer, freeze=freeze_embed)
        self.vocab_dim, self.embed_hidden = pretrain_embed_layer.size()

        # LSTM
        self.rnn_hidden_dim = rnn_hidden_dim // 2 if rnn_bidre else rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.bi_direc = 2 if rnn_bidre else 1
        self.rnn = nn.LSTM(self.embed_hidden, self.rnn_hidden_dim, batch_first=True,
                           num_layers=rnn_layers, bidirectional=rnn_bidre, dropout=rnn_dropout)

        # self.init_rnn_xavier_normal()
        self.rnn_hidden = None

        # dense hidden layers
        self.dense_h_dropout = dense_h_dropout
        self.dense_h0 = self.dense_layer(rnn_hidden_dim, dense_dim, nn.ELU(), dropout=False)

        self.dense_h1 = self.dense_layer(dense_dim, dense_dim, nn.ELU())
        self.dense_h2 = self.dense_layer(dense_dim, dense_dim, nn.ELU())
        self.dense_h3 = self.dense_layer(dense_dim, dense_dim, nn.ELU())

        # output layer
        self.linear = nn.Linear(dense_dim, self.vocab_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, batch_input, sorted_lens):
        # batch_input:   batch, seq_len -
        # sorted_lens:   batch,
        # embed output:   batch, seq_len,embed_dim
        # pack_pad_seq input: batch, Seq_len, *
        # rnn input :  batch, seq_len,input_size; output: seq_len, batch, rnn_hidden_dim
        # pad_pack_seq output: seq_len, batch, *

        word_vec = self.embed(batch_input)
        word_vec = pack_padded_sequence(word_vec, sorted_lens.tolist(), batch_first=True)
        rnn_out, hidden = self.rnn(word_vec)    # hidden : layer*direction, batch, hidden dim
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)

        # dense layers
        unpad = [rnn_out[i, :sorted_lens[i], :] for i in range(len(sorted_lens))]
        decoded = [self.forward_dense(x) for x in unpad]

        # final output
        target_score = [self.log_softmax(x) for x in decoded]
        target_score = torch.cat(target_score)  # batch*seq_len, vocab_size
        return target_score, hidden

    def forward_dense(self, rnn_out):
        # hidden layers
        dense_h0 = self.dense_h0(rnn_out)
        dense_h1 = self.dense_h1(dense_h0)
        dense_h2 = self.dense_h2(dense_h1)
        dense_h3 = self.dense_h3(dense_h2)

        denseout = dense_h1 + dense_h3
        # output layer
        decoded = self.linear(denseout)
        return decoded

    def dense_layer(self, input_dim, output_dim, activation, dropout=True):
        dense = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            activation
        )
        if dropout:
            dense.add_module("Dropout", nn.Dropout(self.dense_h_dropout))
        return dense

    def init_rnn_xavier_normal(self):
        for name, weights in self.rnn.named_parameters():
            weights = weights.view(-1, 1)
            torch.nn.init.xavier_normal_(weights)
            weights.squeeze()

    def init_rnn_hidden(self, batch_size):
        hidden = self.rnn_layers * self.bi_direc
        return torch.zeros(hidden, batch_size, self.rnn_hidden_dim)

    def predict(self, input_word, rnn_hidden=None):
        assert (isinstance(input_word, torch.LongTensor))
        self.train(mode=True)
        word_vec = self.embed(input_word)
        rnn_out, hidden = self.rnn(word_vec, rnn_hidden)
        target = self.forward_dense(rnn_out)
        self.train(mode=False)
        return target, hidden

    def predict_softmax_score(self, input_word, rnn_hidden=None):
        target, hidden = self.predict(input_word, rnn_hidden)
        target_score = F.softmax(target.squeeze(), dim=0)
        return target_score, hidden

    def evaluate_perplexity(self, batch_input, batch_target, sorted_lens, loss_fn):
        # Enter evaluation model s.t. drop/batch norm are off
        self.eval()
        target_score, hidden = self.forward(batch_input, sorted_lens)
        ppl = []
        start = 0
        for length in sorted_lens.tolist():
            score = target_score[start:start+length]
            target = batch_target[start:start+length]
            loss = loss_fn(score, target)
            ppl.append(2**loss.item())
            start += length
        self.train(True)
        return ppl







