from itertools import chain
import torch
import json
from torch import nn
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


class SkipGramNegaSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNegaSampling, self).__init__()
        self.embed_hidden = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.embed_output = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, input_batch, negative_batch):
        # input_batch (N x 2) [x, y]
        # negative_batch (N x k)
        x, y = input_batch
        embed_hidden = self.embed_hidden(x)  # N x 1 x D
        embed_target = self.embed_output(y)  # N x 1 x D
        embed_neg = -self.embed_output(negative_batch)  # N x k x D
        positive_score = embed_target.bmm(embed_hidden.transpose(1, 2)).squeeze(2)  # N x 1
        negative_score = embed_neg.bmm(embed_hidden.transpose(1, 2)).squeeze(2).sum(dim=1, keepdim=True)  # N x 1

        loss = self.log_sigmoid(positive_score) + self.log_sigmoid(negative_score)
        return -torch.mean(loss)

    def predict(self, input_data):
        return self.embed_hidden(input_data)


def construct_weight_table(word_idx_frequency, sample_rates=0.75):
    weights_table = []
    z = 1e-5
    total_words = sum(word_idx_frequency.values())
    for i in word_idx_frequency:
        # the least frequent words will be dropped
        weights_table += [i] * int((word_idx_frequency[i] / total_words) ** sample_rates / z)
    return weights_table  # around 76.8% of vocabs


def get_skipgram_data(data, window_size):
    # data: [ [1,2,3,...], ...] nested list of poetry
    word_target = []
    for x in data:
        for idx in range(len(x)):
            left = [[x[idx], x[i]] for i in range(max(0, idx - window_size), idx)]
            right = [[x[idx], x[i]] for i in range(idx + 1, min(len(x), idx + window_size + 1))]
            word_target += left
            word_target += right
    return word_target  # [[chosen word, target word], ...]


class PoetrySkipGrams(Dataset):
    def __init__(self, data, window_size=3):
        # self.data = data  # nested list of int
        # self.data = get_skipgram_data(self.data, window_size)
        self.window_size = window_size
        self.data = get_skipgram_data(data, window_size)
        self.total_len = len(self.data)
        print(self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        samples = self.data[index]
        return torch.LongTensor(samples)


def get_negative_samples(batch_input, sample_size, weights_table):
    # batch_input [[x,y], [x,y],...]
    batch_samples = []
    for i in batch_input:
        filtered_words = i.data
        negative_samples = []
        while len(negative_samples) < sample_size:
            samples = random.sample(weights_table, k=20 * sample_size)
            negative_samples += [x for x in samples if x not in filtered_words]
        negative_samples = negative_samples[:sample_size]
        batch_samples.append(negative_samples)
    batch_samples = Variable(torch.LongTensor(batch_samples)).contiguous()
    X = batch_input[:, 0].contiguous().view(-1, 1)
    Y = batch_input[:, 1].contiguous().view(-1, 1)

    if use_cuda:
        batch_samples = batch_samples.cuda()
        X, Y = X.cuda(), Y.cuda()
    return (X, Y), batch_samples  # N x sample_size


def prepare_batch_dataloader(data, window_size, batch_size, nega_sample_size, weights_table):
    def SkipGram_collate(batch):
        # batch = torch.cat(batch, dim=0)
        out = default_collate(batch)  # [x=[], y=[]]
        batch_input, negative_samples = get_negative_samples(out, nega_sample_size, weights_table_)  # dump to cuda
        return batch_input, negative_samples

    with open(data) as f:
        data = json.load(f)

    poetry_data = PoetrySkipGrams(data, window_size=window_size)
    nega_sample_size = nega_sample_size
    weights_table_ = weights_table
    skip_gram_dataloader = DataLoader(poetry_data, batch_size,
                                      shuffle=True,
                                      collate_fn=SkipGram_collate)
    return skip_gram_dataloader
