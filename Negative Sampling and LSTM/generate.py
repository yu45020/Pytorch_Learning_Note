import json
import random
import copy
from model_rnn import *
from itertools import cycle, chain
import torch
import random



pretrain_model = "./model_results/rnn_model.pk"
pre_train_weights = "./model_results/fix_embed_model_check_point.pk"
# pre_train_weights = "./model_results/change_embedmodel_check_point.pk"
model = torch.load(pretrain_model)
model.load_state_dict(torch.load(pre_train_weights, map_location='cpu'))
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

non_words = [comma, period, EOP, question_mark]  # , speicla1, question_mark, special2, special3]


def beam_search_forward(model, cache, encoder_output, skip_words=non_words, max_trial=100, remove_punctuation=True):
    caches = []
    trial = 0
    sorted_lens = torch.LongTensor([1])
    for score, init, hidden in cache:
        # target_score, hidden_state = model.predict_softmax_score(init[-1], hidden)
        target_score, hidden_state = model.decoder.predict_softmax_score(init[-1], sorted_lens,
                                                                         encoder_output, hidden,
                                                                         model.attention)

        best_score, index = torch.max(target_score.squeeze(), dim=0)
        while remove_punctuation and index in skip_words and trial < max_trial:
            word = torch.LongTensor([index]).view(1, 1)
            target_score, hidden_state = model.decoder.predict_softmax_score(word, sorted_lens,
                                                                             encoder_output, hidden_state,
                                                                             model.attention)
            best_score, index = torch.max(target_score.squeeze(), dim=0)
            trial += 1
        chosen_word = torch.LongTensor([index]).view(1, 1)
        caches.append([best_score + score, init + [chosen_word], hidden_state])
    return caches


def beam_search(model, init_word, encoder_output, rnn_hidden,
                beam_size=3, text_length=100, skip_words=non_words, remove_punctuation=True):
    sorted_lens = torch.LongTensor([init_word.size(1)])
    target_score, hidden_state = model.decoder.predict_softmax_score(init_word, sorted_lens,
                                                                     encoder_output, rnn_hidden,
                                                                     model.attention)
    target_score = torch.sum(target_score, dim=0).squeeze()
    sorted_score, index = torch.sort(target_score, descending=True)

    # remove chosen words if they are in skip_words list
    # if remove_punctuation:
    #     for i in range(beam_size):
    #         if index[i] in skip_words:
    #             sorted_score.pop(i)
    #             index.pop(i)

    cache = [[sorted_score[i],
              [torch.LongTensor([index[i]]).view(1, 1)],
              hidden_state] for i in range(beam_size)]

    for i in range(text_length):
        cache = beam_search_forward(model, cache, encoder_output, remove_punctuation=remove_punctuation)
    scores = [cache[i][0] for i in range(beam_size)]
    max_id = scores.index(max(scores))
    text = [i.item() for i in cache[max_id][1]]
    return text, cache[max_id][2]  # text, last_hidden_state


#
# init_word = torch.LongTensor(random.sample(list(idx_word), 1)).view(1, 1)
# hidden = None
# beam_results = beam_search(model, init_word, hidden,  beam_size=10, text_length=100)
# scores = [beam_results[i][0] for i in range(len(beam_results))]
# max_id = scores.index(max(scores))
# text = beam_results[max_id][1]
# text = [idx_word[i.item()] for i in text]
# "".join(text)

def random_text(model, beam_search_size=7):
    topic_lengths = model.attention.context_out.in_features - model.attention.context_out.out_features

    topic_count = random.randint(0, topic_lengths)
    random_topic_wors = random.choices(range(len(word_idx)), k=topic_count)
    random_topic_wors += [0] * (topic_lengths - topic_lengths)

    topic = torch.LongTensor(random_topic_wors).view(1,-1)
    topic_lens = torch.LongTensor([topic_count])
    encoder_output, encoder_hidden = model.encoder.forward(topic, topic_lens)

    # title
    # sorted_lens = torch.LongTensor([len(title)])
    # title_words = torch.LongTensor(title).view(1, -1)
    # _, rnn_hidden = model.decoder.forward(title_words, sorted_lens,
    #                                       encoder_output, encoder_hidden,
    #                                       model.attention)

    init_word = torch.randint(0, len(word_idx), size=(1, 1)).long()
    while init_word.item() in non_words:
        init_word = torch.randint(0, len(word_idx), size=(1, 1)).long()

    generated_text = beam_search(model, init_word, encoder_output, encoder_hidden,
                                 beam_size=beam_search_size, text_length=80, remove_punctuation=False)
    return generated_text


def generate_my_poetry(model, title, title_key_word, text_key_words, sentence_len=7, beam_search_size=7,
                       remove_punctuation=True):
    assert not model.training

    topic_lengths = model.attention.context_out.in_features - model.attention.context_out.out_features
    iter_punctuation = cycle([comma, period])

    # topic --> encoder
    key_word_flatted = title_key_word + list(chain.from_iterable(text_key_words))
    key_word_flatted += [0] * (topic_lengths - len(key_word_flatted))
    topics = torch.LongTensor(key_word_flatted).view(1, -1)
    topic_lens = torch.LongTensor([len(key_word_flatted)])
    encoder_output, encoder_hidden = model.encoder.forward(topics, topic_lens)

    # title
    title_words = title + [title_end]
    generated_text = title_words
    sorted_lens = torch.LongTensor([len(title_words)])
    title_words = torch.LongTensor(title_words).view(1, -1)
    _, rnn_hidden = model.decoder.forward(title_words, sorted_lens,
                                          encoder_output, encoder_hidden,
                                          model.attention)

    # main text
    for words in text_key_words:

        text_len = sentence_len - len(words) - 1  # exclude punctuation
        init_word = torch.LongTensor(words).view(1, -1)
        text, rnn_hidden = beam_search(model, init_word, encoder_output, rnn_hidden,
                                       beam_size=beam_search_size, text_length=text_len,
                                       remove_punctuation=remove_punctuation)
        punctuation = [next(iter_punctuation)]
        # punctuation_ = torch.LongTensor(punctuation).view(1,-1)
        # _, rnn_hidden = beam_search(model, punctuation_, encoder_output, rnn_hidden,
        #             beam_size=beam_search_size, text_length=text_len,
        #             remove_punctuation=remove_punctuation)

        generated_text += words + text + punctuation

    return generated_text


title = '柚 子'.split()
title_idx = [word_idx[i] for i in title]
title_key_words = title_idx
key_words = [["夏", "空"], ["彼", "方"], ["天", "神"], ["亂", "漫"], ["魔", "女"], ["夜", "宴"], ["千", "戀"],
             ["萬", "花"]]  # ["天","神"],["亂","漫"],
text_key_words = [list(map(lambda x: word_idx[x], i)) for i in key_words]
out = generate_my_poetry(model, title_idx, title_key_words, text_key_words, sentence_len=7, beam_search_size=3,
                         remove_punctuation=True)
"".join([idx_word[i] for i in out])


out = random_text(model, beam_search_size=10)
"".join([idx_word[i] for i in out])
