from gensim.models import Word2Vec
import json
import logging
import numpy as np

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

word2vec_params = {
    'sg': 1,  # 0 ： CBOW； 1 : skip-gram
    "size": 300,
    "alpha": 0.01,
    "min_alpha": 0.0005,
    'window': 10,
    'min_count': 1,
    'seed': 1,
    "workers": 6,
    "negative": 0,
    "hs": 1,  # 0: negative sampling, 1:hierarchical  softmax
    'compute_loss': True,
    'iter': 50,
    'cbow_mean': 0,
}

with open('./data/poetry.json', 'rb') as f:
    sentences = json.load(f)
    # sentences = sentences[:10]

model = Word2Vec(**word2vec_params)
model.build_vocab(sentences)
trained_word_count, raw_word_count = model.train(sentences, compute_loss=True,
                                                 total_examples=model.corpus_count,
                                                 epochs=model.epochs)

# +++++++++++++++++++++++++++++++++++++
#           Convert to Pytorch Embedding
# -------------------------------------
model = Word2Vec.load("skipgram_model_gensim.pk")


def get_vocab_dict(model):
    word_idx = {key: j.index for key, j in model.wv.vocab.items()}
    idx_word = {i: k for k, i in word_idx.items()}
    return word_idx, idx_word


word_idx, idx_word = get_vocab_dict(model)
weights = model.wv.vectors
embedding_weights = 'word2vec_skipgram_weights.pk'
np.save(embedding_weights, weights)


def rewrite_word_idx_jsons():
    from multiprocessing.dummy import Pool as ThreadPool
    with open('./data/word_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(word_idx, f, ensure_ascii=False)

    # word idx frequency
    with open('./data/idx_word.json', 'w') as f:
        json.dump(idx_word, f)

    # convert the property text into number
    def word_to_idx(sentence):
        return [word_idx[i] for i in sentence]

    p = ThreadPool(4)
    poetry_idx = p.map(word_to_idx, sentences)
    p.close()
    p.join()
    with open('./data/poetry_idx.json', 'w', encoding='utf-8') as f:
        json.dump(poetry_idx, f, ensure_ascii=False)
    print("Done")

    with open("./data/word_frequency.json", 'rb') as f:
        word_freq = json.load(f)
        word_freq = {word_idx[k]:v for k,v in word_freq.items()}

    with open("./data/word_idx_frequency.json", 'w') as f:
        json.dump(word_freq, f)

rewrite_word_idx_jsons()


# +++++++++++++++++++++++++++++++++++++
#           Check result
# -------------------------------------
model_name = 'skipgram_model_gensim.pk'
model.save(model_name)

model.wv.similarity('，', "。")
model.wv.similarity('西', "東")
model.wv.similarity('人', "桃")
model.wv.similarity('天', "地")
model.wv.similarity('日', "月")
model.wv.similarity('君', "人")

model.wv.most_similar("東")
model.wv.most_similar("人")
model.wv.most_similar("汝")
model.wv.most_similar("他")

with open('./data/word_frequency.json', 'rb') as f:
    word_frequency = json.load(f)

from collections import Counter

most_freq = Counter(word_frequency)
frequent_words = most_freq.most_common(150)

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

word_vect = np.array([model.wv.word_vec(i[0], use_norm=False) for i in frequent_words])
word_label = [i[0] for i in frequent_words]

tsne = TSNE(n_components=2, random_state=1, verbose=2, n_iter=15000, perplexity=30)
Y = tsne.fit_transform(word_vect)
x_cord = Y[:, 0]
y_cord = Y[:, 1]

plt.scatter(x_cord, y_cord)
for label, x, y in zip(word_label, x_cord, y_cord):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')



# +++++++++++++++++++++++++++++++++++++
#           Extra
# -------------------------------------

#
# losses = []
# learning_rate = 0.5
# step_size = (0.5 - 0.001) / 10
#
# for i in range(1):
#     end_lr = learning_rate - step_size
#     trained_word_count, raw_word_count = model.train(sentences, compute_loss=True,
#                                                      start_alpha=learning_rate,
#                                                      end_alpha=end_lr,
#                                                      total_examples=model.corpus_count,
#                                                      epochs=1)
#     loss = model.get_latest_training_loss()
#     losses.append(loss)
#     print(i, loss, learning_rate)
#     learning_rate -= step_size
#
#
#
