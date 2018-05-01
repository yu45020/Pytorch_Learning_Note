import json
import os
import time
from sklearn.model_selection import train_test_split
from tensorboard_logger import log_value, configure
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import trange
from model_rnn import *

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

log_file = 'tensorboard_loss'
if not os.path.isdir(log_file):
    os.mkdir(log_file)
try:
    pass
    # configure(log_file, flush_secs=5)
except ValueError:
    pass

# poetry_idx
poetry_data = './data/converted/poetry0.json'
poetry_data = './data/poetry_idx.json'
with open(poetry_data) as f:
    poetry_data = json.load(f)

train, dev = train_test_split(poetry_data, test_size=0.001, train_size=0.99, random_state=4)
print("Length of train {} and length of dev {}".format(len(train), len(dev)))

# prepare pre-trained word embedding and fix its parameters
weights = torch.load("./model_results/SGNS_model_check_point_final2.pt", map_location='cpu')
pre_train_embed_weights = weights['embed_hidden.weight']

max_length = 100
batch_size = 10
train_data = prepare_data_loader(train, max_length, batch_size, shuffle=True)
dev_data = prepare_data_loader(dev, max_length, batch_size=10, shuffle=False)

rnn_hidden_dim = 250
rnn_layer = 3
rnn_bidre = False  # setting it True seems to break the data structure
rnn_dropout = 0
dense_dim = 250
dense_h_dropout = 0.5

model = PoetryRNN(pre_train_embed_weights,
                  rnn_hidden_dim, rnn_layer,
                  rnn_bidre, rnn_dropout,
                  dense_dim, dense_h_dropout,
                  freeze_embed=False)

if os.path.isfile('check_point_model.pkl'):
    model.load_state_dict(torch.load('check_point_model.pkl'))
    print("Load previous training parameters.")

if use_cuda:
    model.cuda()

learning_rate = 5e-3
loss_fn = nn.NLLLoss()
model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
optimizer = optim.Adam(model_params, lr=learning_rate, amsgrad=True,
                       weight_decay=1e-5)

if os.path.isfile("check_point_optim.pkl"):
    optimizer.load_state_dict(torch.load("check_point_optim.pkl"))
    print("Load previous optimizer.")

lr_scheme = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=10, min_lr=1e-7,
                              verbose=True)
counter = 0

t_range = trange(1)
for iteration in t_range:
    print("starting...")
    losses = []
    for batch in train_data:
        batch_x, batch_y, sorted_lens = sort_batches(batch)
        model.zero_grad()
        target_score, hidden = model(batch_x, sorted_lens)        # batch*seq_len x vocab dim
        loss = loss_fn(target_score, batch_y.view(-1))

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        # lr_scheme.step(loss.data[0])
        print("Current batch {}, loss {}".format(counter, loss.item()))
        losses.append(loss.item())
        log_value('Batch_loss', loss.item(), counter)
        counter += 1
        break
    print("One iteration loss {:.3f}".format(np.mean(losses)))



# +++++++++++++++++++++++++++++++++++++
#           Test results
# -------------------------------------

model.load_state_dict(torch.load("./model_results/LSTM_model.pk", map_location='cpu'))

ppl_dev = []
for dev_batch in dev_data:
    batch_x, batch_y, sorted_lens = sort_batches(dev_batch)
    ppl = model.evaluate_perplexity(batch_x, batch_y, sorted_lens, loss_fn)
    ppl_dev.append(ppl)



counter = 0
ppl_train = []
for train_batch in train_data:
    batch_x, batch_y, sorted_lens = sort_batches(train_batch)
    ppl = model.evaluate_perplexity(batch_x, batch_y, sorted_lens, loss_fn)
    ppl_train.append(ppl)
    if counter > 10:
        break
    counter += 1
    print(counter)

import numpy as np
np.mean(list(chain.from_iterable(ppl_train)))
np.mean(list(chain.from_iterable(ppl_dev)))

import types
model.evaluate_perplexity = types.MethodType(evaluate_perplexity, model)