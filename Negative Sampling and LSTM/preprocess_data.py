import json
import glob
from itertools import chain
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
import re
from collections import Counter


patterns = ['（.*）', "{.*}", "《.*》", "\[.*\]", "<.*>", "）", "』", "：", "“.*”",
            '\[', '\」', '；', '》', '（', '）', '/', '`', '、', '：',
            '《', '\*', '-', '=', '{', '}']

def clean_sentence(sentence):
    for x in patterns:
        sentence = re.sub(x, '', sentence)
    return sentence.strip()


def split_poetry_2_list(poetry):
    # one entry in a json file
    # return  a flatten list of words
    text = poetry.get('paragraphs')  # may be []
    if text:
        text = [clean_sentence(x.strip()) for x in text]
        text = list(chain.from_iterable(text))  # flatten list of sentence
        text = ['<SOP>'] + text
        text[-1] = "<EOP>"

        title = poetry.get('title')
        title = "".join(title.split())
        title = clean_sentence(title)
        text = list(title) + text
    return text


def process_data(json_file):
    """
    :param json_file:
    :return: nested list of poetry
    """
    with open(json_file, 'rb') as f:
        data = json.load(f)
    poetry_text = []  # nested list
    word_set = set()
    for poetry in data:
        text = split_poetry_2_list(poetry)  # flatten list
        if text:
            word_set.update(text)
            poetry_text.append(text)
    return poetry_text, word_set


def word_to_idx(word_dict):
    def converter(list_words):
        return [word_dict[w] for w in list_words]

    return converter


if __name__ == "__main__":
    json_location = "./data/json/*.json"
    files = glob.glob(json_location)
    with Pool(processes=4) as pool:
        results = pool.map(process_data, files)

    # nested list of poetry
    poetry_text = [x for y in results for x in y[0]]
    with open("./data/poetry.json", 'w', encoding="utf-8") as f:
        json.dump(poetry_text, f, ensure_ascii=False)

    # word frequency
    word_count = Counter()
    for text in poetry_text:
        word_count.update(text)
    with open('./data/word_frequency.json', 'w', encoding='utf-8') as f:
        json.dump(word_count, f, ensure_ascii=False)

    # unique vocab - number
    vocab = set(word_count)
    dictionary = {k: v for v, k in enumerate(vocab)}
    with open('./data/word_dictionary.json', 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, ensure_ascii=False)

    # word idx frequency
    word_idx_count = {dictionary[k]: v for k, v in word_count.items()}
    with open('./data/word_idx_frequency.json', 'w') as f:
        json.dump(word_idx_count, f)

    # convert the property text into number
    poetry_2_num = word_to_idx(dictionary)
    p = ThreadPool(4)
    poetry_idx = p.map(poetry_2_num, poetry_text)
    p.close()
    p.join()
    with open('./data/poetry_idx.json', 'w', encoding='utf-8') as f:
        json.dump(poetry_idx, f, ensure_ascii=False)
