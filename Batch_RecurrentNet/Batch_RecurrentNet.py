import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.autograd import Variable
import torch.nn as nn
import glob
import os
import string
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from torch.nn.utils import clip_grad_norm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils import data

all_letters = string.ascii_letters + ":.,-\n "
n_letters = len(all_letters)
path_to_files = "./data/names/*.txt"

def load_data(path_to_files):
    files = glob.glob(path_to_files)
    country_name = [os.path.basename(file).split(".")[0] for file in files]
    name_data = []
    for file, file_name in zip(files, country_name):
        with open(file, 'r') as f:
            names = f.readlines()
        name_data += [(name.strip()+"\n", file_name) for name in names]
    return list(set(name_data)), country_name


def prepare_idx(dat):
    to_ix = {k: v for v, k in enumerate(dat)}
    ix_to = {v: k for k, v in to_ix.items()}
    return to_ix, ix_to



def sort_batch_data(dat):
    name, target = dat[0]
    length = dat[1]
    sorted_length, sorted_id = length.sort(dim=0, descending=True)
    sorted_name = name[sorted_id]
    sorted_target = target[sorted_id]
    return Variable(torch.t(sorted_name)), Variable(sorted_target).squeeze(), sorted_length


class NameDataset(data.Dataset):
    def __init__(self, data, char_to_idx, target_to_idx):
        super(NameDataset, self).__init__()
        self.data = data
        self.names = [x[0] for x in data]
        self.targets = [x[1] for x in data]
        self.n_sampel = len(self.data)
        self.length = [len(x[0]) for x in self.data]

        self.char_to_idx = char_to_idx
        self.target_to_idx = target_to_idx

        self.char_tensor = torch.zeros((self.n_sampel, max(self.length))).long()
        self.target_tensor = torch.zeros((self.n_sampel, 1)).long()
        self.process_data()

    def process_data(self):
        for i in range(self.n_sampel):
            name, target = self.data[i]
            self.char_tensor[i, :self.length[i]] = self.convert_to_tensor(name, self.char_to_idx)
            self.target_tensor[i, :] = self.convert_to_tensor([target], self.target_to_idx)

    @staticmethod
    def convert_to_tensor(char, idx):
        return torch.LongTensor([idx[i] for i in char])

    def __getitem__(self, index):
        return [(self.char_tensor[index], self.target_tensor[index]), self.length[index]]

    def __len__(self):
        return len(self.data)


class RecurrentNet(nn.Module):
    def __init__(self, vocab_size, vocab_embed, hidden_dim, target_size, type):
        super(RecurrentNet, self).__init__()
        self.vocab_embed = vocab_embed
        self.hidden_dim = hidden_dim

        self.embed = nn.Embedding(vocab_size, vocab_embed)
        if type =='LSTM':
            self.rnn = nn.LSTM(vocab_embed, hidden_dim//2, num_layers=1, bias=True, bidirectional=True)
        elif type == 'GRU':
            self.rnn = nn.GRU(vocab_embed, hidden_dim//2, num_layers=1, bias=True, bidirectional=True)
        else:
            raise SyntaxError("Type Error. Either LSTM or GRU")
        self.type = type
        self.char_2_tag = nn.Linear(hidden_dim, target_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size):
        if self.type == "LSTM":
            hidden_layers = (Variable(torch.zeros((2, batch_size, self.hidden_dim // 2))),
                             Variable(torch.zeros((2, batch_size, self.hidden_dim // 2))))
        elif self.type == 'GRU':
            hidden_layers = Variable(torch.zeros((2, batch_size, self.hidden_dim // 2)))
        return hidden_layers

    def forward(self, name_tensor, sorted_length):
        embed_name = self.embed(name_tensor)
        packed_name = pack_padded_sequence(embed_name, sorted_length.tolist())
        char_gru, char_hidden = self.rnn(packed_name, self.hidden)
        pad_char, _ = pad_packed_sequence(char_gru)
        target_space = self.char_2_tag(pad_char)
        target_space = target_space.sum(dim=0)
        target_score = self.softmax(target_space)
        return target_score


def split_data(data, test_size=0.3, train_size=0.7, random_state=1):
    splitter = StratifiedShuffleSplit(test_size=test_size,
                                      train_size=train_size,
                                      random_state=random_state)
    y_true = [x[1] for x in data]
    for train_index, test_index in splitter.split(data, y_true):
        train = [data[i] for i in train_index]
        test = [data[i] for i in test_index]
    return train, test


name_data, countries = load_data(path_to_files)
train, test = split_data(name_data, test_size=0.3, train_size=0.7, random_state=1)

char_to_ix, ix_to_char = prepare_idx(all_letters)
target_to_ix, ix_to_target = prepare_idx(countries)
vocab_size = len(char_to_ix)
target_size = len(target_to_ix)

vocab_embed = 80
hidden_dim = 60
iterations = 1
batch_size = 256



train_set = NameDataset(train, char_to_ix, target_to_ix)
train_data = DataLoader(train_set, batch_size, shuffle=True)
test_set = NameDataset(test, char_to_ix, target_to_ix)
test_data = DataLoader(test_set, len(test), shuffle=False)

model = RecurrentNet(vocab_size, vocab_embed, hidden_dim, target_size, "LSTM")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.001)
optim_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, verbose=True, patience=4)
loss_cache = []
for i in range(iterations):
    total_loss = []
    for batch_data in train_data:
        name, target, length = sort_batch_data(batch_data)
        batch_size = len(length)
        model.zero_grad()
        model.hidden = model.init_hidden(batch_size)
        target_score = model(name, length)
        loss = loss_fn(target_score, target)
        loss.backward()
        clip_grad_norm(model.parameters(), max_norm=10)
        optimizer.step()
        total_loss.append(loss.data[0] * batch_size)
    loss_avg = sum(total_loss) / len(total_loss)
    optim_scheduler.step(loss_avg)
    print("Epoch {} : {:.5f}".format(i, loss_avg))
    loss_cache.append(loss_avg)

plt.plot(loss_cache)

def evaluation(data):
    # post-process data from data loader
    target_pred = []
    target_true = []
    for batch_data in data:
        name, target, length = sort_batch_data(batch_data)
        batch_size = len(length)
        model.hidden = model.init_hidden(batch_size)
        target_score = model(name, length)
        score, category = torch.max(target_score, dim=1)
        target_pred += category.data.tolist()
        target_true += target.data.tolist()
    return target_pred, target_true

target_pred, target_true = evaluation(train_data)
accuracy_score(target_true, target_pred)

report = classification_report(target_true, target_pred, target_names=list(target_to_ix))
print(report)

error_matrix = confusion_matrix(target_true, target_pred)
print(list(target_to_ix))
print(error_matrix)

y_true = [ix_to_target[i] for i in target_true]
y_pred = [ix_to_target[i] for i in target_pred]