import torch
import json

from cytoolz.itertoolz import itemgetter

from model_negative_sampling import *
import numpy as np

with open('./data/word_dictionary.json', 'rb') as f:
    dictionary = json.load(f)
    idx_word = {v: k for k, v in dictionary.items()}

with open('./data/word_idx_frequency.json', 'r') as f:
    word_idx_frequency = json.load(f)
    word_idx_frequency = {int(k): v for k, v in word_idx_frequency.items()}

vocab_size = len(word_idx_frequency)
embed_dim = 300

SGNS = SkipGramNegaSampling(vocab_size, embed_dim)


trained_weights = './model_results/SGNS_model_check_point_final2.pt'
SGNS.load_state_dict(torch.load(trained_weights, map_location='cpu'))

# load from LSTM re-trained weights
LSTM_w = torch.load('./model_results/LSTM_model.pk', map_location='cpu')
SGNS.embed_hidden.weight.data = LSTM_w['embed.weight']

SGNS.eval()

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

word_counter = Counter(word_idx_frequency)
word = word_counter.most_common()[:150]
word = [x[0] for x in word]

# word = [dictionary[i] for i in ['東','西']]
word_vect = SGNS.embed_hidden(torch.LongTensor(word))

word_vect = word_vect.data.numpy()
word_label = [idx_word[i] for i in word]

tsne = TSNE(n_components=2, random_state=1, verbose=2, n_iter=15000, perplexity=50)
Y = tsne.fit_transform(word_vect)
x_cord = Y[:, 0]
y_cord = Y[:, 1]

plt.scatter(x_cord, y_cord)
for label, x, y in zip(word_label, x_cord, y_cord):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


# +++++++++++++++++++++++++++++++++++++
#           Find Similarity
# -------------------------------------

sop = dictionary['<SOP>']
eop = dictionary['<EOP>']

def compare_two_words(word_a_id, word_b_id, model=SGNS):
    cos_similar = torch.nn.CosineSimilarity(dim=1)
    word_a = torch.LongTensor([word_a_id]).view(1, -1)
    word_b = torch.LongTensor([word_b_id]).view(1, -1)
    vect_a = model.predict(word_a).view(1, -1)
    vect_b = model.predict(word_b).view(1, -1)
    similarity = cos_similar(vect_a, vect_b)
    return similarity


def find_most_similar_words(base_word_id, idx_word, model=SGNS):
    cosine_collection = []
    word_2_vect = model.embed_hidden.weight.data
    vocab_size = word_2_vect.size(0)
    for i in range(vocab_size):
        similarity = compare_two_words(base_word_id, i, model=SGNS)
        cosine_collection.append((i, similarity.item()))

    cosine_collection = [[idx_word[x[0]], x[1]] for x in cosine_collection]  # change to characters
    cosine_collection.sort(key=itemgetter(1), reverse=True)
    return cosine_collection


people_id = dictionary['人']
most_similar_words = find_most_similar_words(people_id,idx_word)
most_similar_words[:10]

stop = dictionary['。']
compare_two_words(stop, eop)

sky = dictionary['天']
earth = dictionary['地']
compare_two_words(sky, earth)


