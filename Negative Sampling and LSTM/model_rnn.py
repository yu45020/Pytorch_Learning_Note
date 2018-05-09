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
    def __init__(self, data, max_topic_counts=20, max_poetry_length=100):
        # when chunk size = 120, evenly divide; = 259, leave one out
        # most poetries have length around 40 - 80
        # data is nested list of word idx
        assert any(isinstance(i, list) for i in data)

        # topic words
        topics = [i[0] for i in data]
        self.topic_lens = torch.LongTensor([min(len(x), max_topic_counts) for x in topics])

        # poetry text
        data = [i[1] for i in data]
        self.lengths = torch.LongTensor([min(len(x), max_poetry_length) - 1 for x in data])
        self.lens = len(self.lengths)

        # pad data
        max_len = min(max(self.lengths), max_poetry_length)

        self.topics = torch.zeros((self.lens, max_topic_counts)).long()
        self.data = torch.zeros((self.lens, max_len)).long()
        self.target = torch.zeros((self.lens, max_len)).long()
        for i in range(self.lens):
            TL = min(self.topic_lens[i], max_topic_counts)
            self.topics[i, :TL] = torch.LongTensor(topics[i][:TL])

            L = min(self.lengths[i], max_poetry_length)
            self.data[i, :L] = torch.LongTensor(data[i][:L])
            self.target[i, :L] = torch.LongTensor(data[i][1:(L + 1)])
        if use_cuda:
            self.topics = self.topics.cuda()
            self.topic_lens = self.topic_lens.cuda()
            self.data = self.data.cuda()
            self.target = self.target.cuda()
            self.lengths = self.lengths.cuda()

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        out = (
            self.topics[index, :], self.topic_lens[index], self.data[index, :], self.target[index, :],
            self.lengths[index])
        return out


def prepare_data_loader(data, max_topic_len, max_length, batch_size, shuffle):
    dataset = PoetryRNNData(data, max_topic_counts=max_topic_len, max_poetry_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def sort_batches(batches):
    topics, topics_len, x, y, lengths = batches
    # topic key words
    sorted_topic_lens, sorted_topic_idx = topics_len.sort(dim=0, descending=True)
    sorted_topics = topics.index_select(dim=0, index=sorted_topic_idx.squeeze())
    sorted_topics = sorted_topics.contiguous()
    pad_topic_lens = sorted_topic_lens.squeeze()

    # poetry text
    sorted_lens, sorted_idx = lengths.sort(dim=0, descending=True)

    sorted_x = x.index_select(dim=0, index=sorted_idx.squeeze())  # x[sorted_idx, :]
    sorted_y = y.index_select(dim=0, index=sorted_idx.squeeze())  # y[sorted_idx, :]

    pad_len = sorted_lens.squeeze()
    unpad_y = [sorted_y[i, :pad_len[i]] for i in range(len(pad_len))]
    unpad_y = torch.cat(unpad_y)

    sorted_x = sorted_x.contiguous()
    unpad_y = unpad_y.contiguous()
    # if use_cuda:
    #    sorted_x = sorted_x.cuda()
    #    unpad_y = unpad_y.cuda()

    out = sorted_topics, pad_topic_lens, sorted_x, unpad_y, pad_len
    return out


# +++++++++++++++++++++++++++++++++++++
#           Prepare RNN model      
# -------------------------------------

class PoetryEncoder(nn.Module):
    def __init__(self, encoder_embed,
                 rnn_hidden_dim, rnn_layers,
                 freeze_embed=False):
        super(PoetryEncoder, self).__init__()
        self.embed = nn.Embedding.from_pretrained(encoder_embed, freeze=freeze_embed)
        self.vocab_dim, self.embed_hidden = encoder_embed.size()

        # GRU: the output looks similar to hidden state of standard RNN
        self.rnn_hidden_dim = rnn_hidden_dim // 2
        self.rnn_layers = rnn_layers
        self.rnn = nn.GRU(self.embed_hidden, self.rnn_hidden_dim, batch_first=True,
                          num_layers=self.rnn_layers, bidirectional=True)

        # attention

    def forward(self, batch_input, sorted_lens):
        # batch_input:   batch, seq_len -
        # sorted_lens:   batch,
        # embed output:   batch, seq_len,embed_dim
        # pack_pad_seq input: batch, Seq_len, *
        # rnn input :  batch, seq_len,input_size; output: seq_len, batch, rnn_hidden_dim
        # pad_pack_seq output: seq_len, batch, *

        word_vec = self.embed(batch_input)
        word_vec = pack_padded_sequence(word_vec, sorted_lens.tolist(), batch_first=True)
        rnn_out, hidden = self.rnn(word_vec)  # hidden : layer*direction, batch, hidden dim
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=batch_input.size(1))
        hidden = self.reshape_bidirec_hidden(hidden)
        return rnn_out, hidden

    def merge_bidirec_hidden(self, hidden_state):
        # in bi-directions layers, each layer contains 2 directions of hidden states, so
        # take their average for each layer
        h = hidden_state
        h = torch.cat(list(((h[i] + h[i + 1]) / 2).unsqueeze(0) for i in range(0, self.rnn_layers * 2, 2)))
        # c = torch.cat(list(((c[i] + c[i + 1])/2).unsqueeze(0) for i in range(0, self.rnn_layers * 2, 2)))
        return (h, h)

    def reshape_bidirec_hidden(self, hidden_state):
        h = hidden_state
        num_layers, batch, hidden_size = h.size()
        h = h.reshape(num_layers // 2, batch, -1)

        # c = torch.zeros_like(h)
        return (h, h)


class KeywordAttention(nn.Module):
    def __init__(self, encoder_topic_len, encoder_hidden_dim, decoder_embed_hidden, attention_dropout=0.1):
        super(KeywordAttention, self).__init__()
        self.atten_weights = nn.Linear(encoder_hidden_dim, decoder_embed_hidden)
        self.softmax = nn.Softmax(dim=2)
        self.context_out = nn.Linear(encoder_topic_len + decoder_embed_hidden, decoder_embed_hidden)
        self.dropout = nn.Dropout(attention_dropout)
        self.activation_out = nn.SELU()

    def forward(self, decoder_input, encoder_output):
        # decoder_input: batch, seq_len, embedding_hidden
        # rnn_output: batch, seq_len, rnn_hidden
        # encoder_output: batch, topic_len, rnn_hidden
        # context_state = encoder_hidden[0].t()  # --> batch, num_layer, hidden_dim
        context_state = self.dropout(encoder_output)
        attention = self.atten_weights(context_state).transpose(1, 2)  # --> batch, decoder_embed_hidden, topic_len

        attention_w = decoder_input.bmm(attention)  # batch, seq_len, topic_len
        attention = self.softmax(attention_w)

        context_concat = torch.cat([decoder_input, attention], dim=2)  # batch, seq_len, topic_len+embed_hidden
        out = self.context_out(context_concat)  # batch, seq_len, embed_hidden
        out = self.activation_out(out)
        return out


class PoetryDecoder(nn.Module):
    def __init__(self, decoder_embed,
                 rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                 dense_dim, dense_h_dropout,
                 freeze_embed=False):
        super(PoetryDecoder, self).__init__()

        # pre-trained word embedding
        self.embed = nn.Embedding.from_pretrained(decoder_embed, freeze=freeze_embed)
        self.vocab_dim, self.embed_hidden = decoder_embed.size()

        # LSTM
        self.rnn_hidden_dim = rnn_hidden_dim // 2 if rnn_bidre else rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.bi_direc = 2 if rnn_bidre else 1
        self.rnn = nn.LSTM(self.embed_hidden, self.rnn_hidden_dim, batch_first=True,
                           num_layers=rnn_layers, bidirectional=rnn_bidre, dropout=rnn_dropout)

        # self.init_rnn_xavier_normal()

        # dense hidden layers
        self.dense_h_dropout = dense_h_dropout
        self.dense_h0 = self.dense_layer(rnn_hidden_dim, dense_dim, nn.SELU(), dropout=True)

        self.dense_h1 = self.dense_layer(dense_dim, dense_dim, nn.SELU())
        self.dense_h2 = self.dense_layer(dense_dim, dense_dim, nn.SELU())
        self.dense_h3 = self.dense_layer(dense_dim, dense_dim, nn.SELU())

        # output layer
        self.output_linear = nn.Linear(dense_dim, self.vocab_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_(self, batch_input, sorted_lens, encoder_output, rnn_hidden, attention):
        # batch_input:   batch, seq_len -
        # sorted_lens:   batch,
        # encoder_output: batch, topic_len, hidden_dim
        # rnn_hidden: (h,c), h: num_layers, batch, hidden_dim (concat)
        # embed output:   batch, seq_len,embed_dim
        # pack_pad_seq input: batch, Seq_len, *
        # rnn input :  batch, seq_len,input_size; output: seq_len, batch, rnn_hidden_dim
        # pad_pack_seq output: seq_len, batch, *

        word_vec = self.embed(batch_input)
        # mask out zeros for padded 0s
        word_vec = self.mask_zeros(word_vec, sorted_lens)
        # attention
        word_vec = attention.forward(word_vec, encoder_output)

        word_vec = pack_padded_sequence(word_vec, sorted_lens.tolist(), batch_first=True)
        rnn_out, hidden = self.rnn(word_vec, rnn_hidden)  # hidden : layer*direction, batch, hidden dim
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True, total_length=batch_input.size(1))

        # attention

        # dense layers
        unpad = [rnn_out[i, :sorted_lens[i], :] for i in range(len(sorted_lens))]
        decoded = [self.forward_dense(x) for x in unpad]

        # final output
        return decoded, hidden

    def forward(self, batch_input, sorted_lens, encoder_output, rnn_hidden, attention):
        decoded, hidden = self.forward_(batch_input, sorted_lens, encoder_output, rnn_hidden, attention)
        target_score = [self.log_softmax(x) for x in decoded]
        target_score = torch.cat(target_score)  # batch*seq_len, vocab_size
        return target_score, hidden

    def forward_dense(self, rnn_out):
        # hidden layers
        dense_h0 = self.dense_h0(rnn_out)
        # dense_h1 = self.dense_h1(dense_h0)
        # dense_h2 = self.dense_h2(dense_h1)
        # dense_h3 = self.dense_h3(dense_h2)
        #
        # denseout = dense_h1 + dense_h3
        # output layer
        decoded = self.output_linear(dense_h0)
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

    @staticmethod
    def mask_zeros(word_vec, sorted_lens):
        # word_vec: batch, seq_len, embed_dim
        # Each example has different lengths, but the padded 0s have value after the embedding layer, so mask them 0

        for i in range(len(sorted_lens)):
            if sorted_lens[i] < word_vec[i].size(0):
                word_vec[i, sorted_lens[i]:, :] = 0
        return word_vec

    def init_rnn_xavier_normal(self):
        for name, weights in self.rnn.named_parameters():
            weights = weights.view(-1, 1)
            torch.nn.init.xavier_normal_(weights)
            weights.squeeze()

    def predict_softmax_score(self,batch_input, sorted_lens, encoder_output, rnn_hidden, attention):
        assert not self.training
        decoded, hidden = self.forward_(batch_input, sorted_lens, encoder_output, rnn_hidden, attention)
        target_score = [F.softmax(x, dim=1) for x in decoded]
        target_score = torch.cat(target_score)  # batch*seq_len, vocab_size
        return target_score, hidden

class PoetryRNN(nn.Module):
    def __init__(self, encoder_embed, encoder_topic_len,
                 decoder_embed, rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                 dense_dim, dense_h_dropout,
                 freeze_embed=True):
        super(PoetryRNN, self).__init__()

        self.encoder = PoetryEncoder(encoder_embed,
                                     rnn_hidden_dim, rnn_layers,
                                     freeze_embed)
        self.attention = KeywordAttention(encoder_topic_len, rnn_hidden_dim, encoder_embed.size(1))
        self.decoder = PoetryDecoder(decoder_embed,
                                     rnn_hidden_dim, rnn_layers, rnn_bidre, rnn_dropout,
                                     dense_dim, dense_h_dropout,
                                     freeze_embed)

    def forward(self, batch_topic, topic_lens, batch_input, sorted_lens):
        encoded_output, encoded_hidden = self.encoder.forward(batch_topic, topic_lens)
        target_score, hidden = self.decoder.forward(batch_input, sorted_lens, encoded_output, encoded_hidden,
                                                    self.attention)
        return target_score, hidden

    def predict(self, batch_topic, topic_lens, batch_input, sorted_lens):
        assert not self.training
        decoded, hidden = self.forward_(batch_topic, topic_lens, batch_input, sorted_lens)
        return decoded, hidden

    def predict_softmax_score(self, batch_topic, topic_lens, batch_input, sorted_lens):
        assert not self.training
        decoded, hidden = self.forward(batch_topic, topic_lens, batch_input, sorted_lens)
        target_score = [F.softmax(x, dim=1) for x in decoded]
        target_score = torch.cat(target_score)
        return target_score, hidden



    def evaluate_perplexity(self, batch_topic, topic_lens,
                            batch_input, sorted_lens,
                            batch_target, loss_fn):
        # Enter evaluation model s.t. drop/batch norm are off
        assert not self.training
        target_score, hidden = self.forward(batch_topic, topic_lens, batch_input, sorted_lens)
        ppl = []
        start = 0
        for length in sorted_lens.tolist():
            score = target_score[start:start + length]
            target = batch_target[start:start + length]
            loss = loss_fn(score, target)
            ppl.append(2 ** loss.item())
            start += length
        return ppl


