import json
import random
import copy
from model_rnn import *
from itertools import cycle,chain
import torch

weights = torch.load("./model_results/SGNS_model_check_point_final2.pt", map_location='cpu')
pre_train_embed_weights = weights['embed_hidden.weight']

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

model_weights = torch.load('./model_results/LSTM_model.pk', map_location='cpu')
model.load_state_dict(state_dict=model_weights)

model = model.eval()

word_idx_path = './data/word_dictionary.json'

with open(word_idx_path, 'rb') as f:
    word_idx = json.load(f)
    idx_word = {k: v for v, k in word_idx.items()}

title_end = word_idx['<SOP>']
comma = word_idx['，']
period = word_idx['。']
question_mark = word_idx['？']
EOP = word_idx['<EOP>']
speicla1 = word_idx['」']
special2 = word_idx['；']
special3 = word_idx['》']
non_words = [comma, period, EOP, speicla1, question_mark, special2, special3]


def generate_word(model, init_word, rnn_hidden=None, sample_max_prob=False):
    target_score, rnn_hidden = model.predict_softmax_score(init_word, rnn_hidden)
    if sample_max_prob:
        init_word = torch.argmax(target_score)
    else:
        init_word = torch.multinomial(target_score, 1)
    return init_word.view(1, 1), rnn_hidden


def generate_seq_text(model, init_word=None, text_length=200, sample_max_prob=False):
    if init_word is None:
        word_list = [x for x in list(idx_word) if x != "<EOP>"]
        init_word = torch.LongTensor(random.sample(word_list, 1)).view(1, 1)
    else:
        assert (isinstance(init_word, torch.LongTensor))
    generate_text = [init_word.item()]
    rnn_hidden = None
    while len(generate_text) < text_length and init_word.item() != EOP:
        init_word, rnn_hidden = generate_word(model, init_word, rnn_hidden, sample_max_prob)
        generate_text.append(init_word.item())
    return generate_text


def generate_my_text(model, title_words, init_words, sample_max_prob=False):
    # all input words are flatten lists of int
    rnn_hidden = None
    titles_id = title_words + [title_end]
    init_words_idx = copy.copy(init_words)
    text_out = copy.copy(titles_id)

    # get rnn_hidden values for the given title
    while titles_id:
        init_word = titles_id.pop(0)
        init_word = torch.LongTensor([init_word]).view(1, 1)
        _, rnn_hidden = generate_word(model, init_word, rnn_hidden=rnn_hidden, sample_max_prob=sample_max_prob)

    # body
    iter_punctuation = cycle([comma, period])  # repeat the two values forever
    while init_words_idx:
        one_line_text = []
        # fix beginning words for all sentences
        init_word = init_words_idx.pop(0)
        for i in init_word:
            one_line_text.append(i)
            init_word = torch.LongTensor([i]).view(1, 1)
            _, rnn_hidden = generate_word(model, init_word, rnn_hidden=rnn_hidden,
                                           sample_max_prob=sample_max_prob)

        # sample the rest words
        while len(one_line_text) < 7:
            init_word_, rnn_hidden_ = generate_word(model, init_word, rnn_hidden=rnn_hidden,
                                                    sample_max_prob=sample_max_prob)
            if init_word_ not in non_words:
                init_word, rnn_hidden = init_word_, rnn_hidden_
                one_line_text.append(init_word.item())
            else:
                rnn_hidden = rnn_hidden_

        punctuation = next(iter_punctuation)
        one_line_text.append(punctuation)
        text_out += one_line_text
        punctuation = torch.LongTensor([punctuation]).view(1, 1)
        _, rnn_hidden = generate_word(model, punctuation, rnn_hidden=rnn_hidden,
                                      sample_max_prob=sample_max_prob)

    return text_out

out = generate_seq_text(model, init_word=None, text_length=200, sample_max_prob=False)
''.join([idx_word[i] for i in out])


title = '柚 子 水'.split()
title_idx = [word_idx[i] for i in title]

init_words = [["夏","空"],["彼","方"],["魔","女"],["夜","宴"],["千","戀"], ["萬","花"]] # ["天","神"],["亂","漫"],
init_words_idx = [list(map(lambda x: word_idx[x], i)) for i in init_words]

out = generate_my_text(model, title_idx, init_words_idx, sample_max_prob=False)
"".join([idx_word[i] for i in out])
