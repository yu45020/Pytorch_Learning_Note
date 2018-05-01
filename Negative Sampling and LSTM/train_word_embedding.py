#   torch.__version__  == 0.4. dev
import glob
import json
import os
import time

import numpy as np
from tensorboard_logger import configure, log_value
from torch import optim
from tqdm import tqdm

from model_negative_sampling import *

if not os.path.exists("skipgram_loss"):
    os.mkdir('skipgram_loss')

configure("skipgram_loss")
SGNS_pretrain_model = None  # path to pickled file
SGNS_pretrain_optim = None  # path to pickled file

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor


with open('./data/word_idx_frequency.json', 'r') as f:
    word_idx_frequency = json.load(f)
    word_idx_frequency = {int(k): v for k, v in word_idx_frequency.items()}

weights_table = construct_weight_table(word_idx_frequency)  # LongTensor, one dimension

vocab_size = len(word_idx_frequency)
embed_dim = 300

SGNS = SkipGramNegaSampling(vocab_size, embed_dim)
optimer = optim.SparseAdam(SGNS.parameters(), lr=1e-2)

if SGNS_pretrain_model:
    SGNS.load_state_dict(torch.load('SGNS_pre_trained.pt'))
    print("Load pre-trained weights")

if SGNS_pretrain_optim:
    optimer.load_state_dict(torch.load('SGNS_optimizer_pre_trained.pt.pt'))
    print("Load pre-trained optimizer")


if use_cuda:
    SGNS = SGNS.cuda()
    print("Dump to Cuda")

files = sorted(glob.glob('./converted/*.json'))

window_size = 2
batch_size = 1024 * 20
nega_sample_size = 5
counter = 0
losses = []
start = time.time()
for iteration in tqdm(range(1)):
    file_counter = 0
    for file in files:
        poetry_data = prepare_batch_dataloader(file, window_size, batch_size,
                                               nega_sample_size, weights_table)
        start_t = time.time()
        losses = []
        for batch_data in poetry_data:
            # batch_input: (x, y) ; negative_samples: (batch_size,  nega_sample_size)
            batch_input, negative_samples = batch_data
            SGNS.zero_grad()
            loss = SGNS.forward(batch_input, negative_samples)
            loss.backward()
            optimer.step()
            counter += 1
            losses.append(loss.item())
            msg = "Current batch {} , batch loss : {:.8f}, time:{:.2f}".format(counter,
                                                                               loss.data[0],
                                                                               time.time() - start_t)
            print(msg)
            log_value('batch_loss', loss.item(), counter)
        print("File {} loss {:.5f}, time{:.2f}".format(os.path.basename(file), np.mean(losses), time.time() - start_t))

        if file_counter % 77 == 0:
            print("Saving check_point up to file {}".format(file_counter))
            embed_file = 'SGNS_model_check_poin' + str(file_counter) + '.pk'
            torch.save(SGNS.state_dict(), embed_file)

            optim_file = 'SGNS_optim_check_poin' + str(file_counter) + '.pk'
            torch.save(optimer.state_dict(), optim_file)
        file_counter += 1
