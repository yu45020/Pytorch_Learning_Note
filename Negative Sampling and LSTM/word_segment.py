# from  https://github.com/DevinZ1993/Chinese-Poetry-Generation/blob/master/segment.py
# RNN-based Poem Generator by
import jieba
import os
import json
from opencc import OpenCC


sxhy_raw = 'shixuehanying.txt'
sxhy_path = 'sxhy_dict.txt'

def _gen_sxhy_dict():
    sxhy_dict = dict()
    with open(sxhy_raw, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        while line:
            if line.startswith('<begin>'):
                tag = line.split('\t')[2]
            elif not line.startswith('<end>'):
                toks = line.split('\t')
                if len(toks) == 3:
                    toks = toks[2].split(' ')
                    tok_list = []
                    for tok in toks:
                        if len(tok) < 4:
                            tok_list.append(tok)
                        else:
                            tok_list.extend(jieba.lcut(tok, HMM=True))
                    for tok in tok_list:
                        sxhy_dict[tok] = tag
            line = f.readline().strip()
    with open(sxhy_path, 'w', encoding='utf-8') as f:
        for word in sxhy_dict:
            f.write(word+'\n')

if not os.path.exists(sxhy_path):
    _gen_sxhy_dict()

with open(sxhy_path, 'r', encoding='utf-8') as f:
    word_dict = f.readlines()
    word_dict = [i.strip() for i in word_dict]

openCC_converter = OpenCC('s2t')

word_dict_tc = [openCC_converter.convert(i) for i in word_dict]


with open('sxhy_traditional.txt', 'w', encoding='utf-8') as f:
    for word in word_dict_tc:
        f.write(word + '\n')


jieba.load_userdict('sxhy_traditional.txt')
jieba.lcut("只應歲晚凌霜操，併託松篁待主人")



def get_sxhy_dict():
    sxhy_dict = set()
    with open(sxhy_path, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            sxhy_dict.add(line.strip())
            line = f.readline()
    return sxhy_dict
