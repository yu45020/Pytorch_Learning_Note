import json
import os
import time
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from model_rnn import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


# poetry_idx

poetry_data = './data/sample.json'
with open(poetry_data, 'rb') as f:
    poetry_data = json.load(f)

poetry_data = [x for x in poetry_data if len(x[0])>1]  # remove empty data

train, dev = train_test_split(poetry_data, test_size=0.001, train_size=0.99, random_state=4)
print("Length of train {} and length of dev {}".format(len(train), len(dev)))


weights = np.load('./model_results/word2vec_skipgram_weights.pk.npy')
embed_weights = torch.from_numpy(weights)

max_topic_len = 20
max_length = 100
batch_size = 256
train_data = prepare_data_loader(train,max_topic_len, max_length, batch_size, shuffle=True)
dev_data = prepare_data_loader(dev, max_topic_len, max_length, batch_size=10, shuffle=False)

rnn_hidden_dim = 300
rnn_layer = 2
rnn_bidre = False  # setting it True seems to break the data structure
rnn_dropout = 0
dense_dim = 300
dense_h_dropout = 0.5

model = PoetryRNN(embed_weights, max_topic_len,
                  embed_weights,rnn_hidden_dim, rnn_layer,
                  rnn_bidre, rnn_dropout,
                  dense_dim, dense_h_dropout,
                  freeze_embed=True)

model_check_point = "./model_results/fix_embed_model_check_point.pk"
if os.path.isfile(model_check_point):
    model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
    model.train(True)
    print("Load previous training parameters.")

if use_cuda:
    model = model.cuda()
    print("Dump to cuda")

model.train(True)

learning_rate = 5e-3
loss_fn = nn.NLLLoss()
model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(model_params, lr=learning_rate, amsgrad=True,weight_decay=0)

if os.path.isfile("check_point_optim.pkl"):
    optimizer.load_state_dict(torch.load("check_point_optim.pkl"))
    print("Load previous optimizer.")

lr_scheme = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10, min_lr=1e-7,
                              verbose=True)
counter = 0
all_loss = []
t_range = trange(1)
for iteration in t_range:
    print("starting...")
    losses = []
    for batch in train_data:
        batch_topic, topic_lens, batch_x, batch_y, sorted_lens = sort_batches(batch)
        model.zero_grad()
        target_score, hidden = model(batch_topic, topic_lens, batch_x, sorted_lens)        # batch*seq_len x vocab dim
        loss = loss_fn(target_score, batch_y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        # lr_scheme.step(loss.data[0])
        print("Current batch {}, loss {}".format(counter, loss.item()))
        losses.append(loss.item())
        counter += 1
        all_loss.append(loss.item())

    one_ite_loss = np.mean(losses)
    lr_scheme.step(one_ite_loss)
    print("One iteration loss {:.3f}".format(one_ite_loss))



torch.save(model.state_dict(), 'model_check_point.pk')
torch.save(model, 'rnn_model.pk')
torch.save(optimizer.state_dict(), 'optimizer_check_point.pk')
torch.save(all_loss, 'all_loss_check_point.pk')


import torch
a = torch.randn(10,5,2)
lengths = [5,4,3,2,1,1,2,3,4,5]
for i in range(len(lengths)):
    if lengths[i] < a[i,:].size(0):
        a[i, lengths[i]:, :] = 0


