import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idx = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idx)
    return autograd.Variable(tensor)


training_data = [
    ("The the apple dim A".split(), ["DET", "V", "NN", 'V',"NN"]),
    ("Everybody read book AAA BBB CCC".split(), ["NN", "V", "NN", "NN", "NN","NN"])
]

sentences = [dat[0] for dat in training_data]
vocabs = set([item for sublist in sentences for item in sublist])

word_to_ix = {w: i for i, w in enumerate(vocabs)}
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
# 26 lower case letters
char_to_ix = {chr(c): i for i, c in enumerate(range(97, 123))}

prepare_sequence(training_data[0][0], word_to_ix)
a = [prepare_sequence(w.lower(), char_to_ix) for w in training_data[0][0]]


class LSTMTagger(nn.Module):
    def __init__(self, vocab_size, word_embed, char_embed, max_words, max_chars, word_hidden_dim, char_hidden_dim,
                 tagset_size):
        super(LSTMTagger, self).__init__()
        self.max_words = max_words
        self.max_chars = max_chars

        self.word_hidden_dim = word_hidden_dim
        self.char_hidden_dim = char_hidden_dim

        self.char_embeddings = nn.Embedding(26, char_embed)
        self.lstm_char = nn.LSTM(char_embed * max_chars, char_hidden_dim)

        self.char_2_word_relu = nn.ReLU()
        self.char_hidden_2_word = nn.Linear(char_hidden_dim, word_hidden_dim)
        self.char_hidden = self.init_hidden()

        self.word_embed = word_embed
        self.word_embeddings = nn.Embedding(vocab_size, word_embed)
        self.lstm_word = nn.LSTM(word_embed + char_embed * max_chars, word_hidden_dim)
        self.word_2_tag = nn.Linear(word_hidden_dim, tagset_size)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros((1, 1, self.char_hidden_dim))),
                autograd.Variable(torch.zeros((1, 1, self.char_hidden_dim))))

    def pad_chars(self, char_tensor):
        w, h = char_tensor.size()
        pad = autograd.Variable(torch.zeros((self.max_chars, h)))
        out = torch.cat([char_tensor, pad], dim=0)
        out = out[:self.max_chars]
        return out.view(1, -1)

    def prepare_embed_char(self, chars):
        embeds_char = [self.char_embeddings(k) for k in chars]
        embeds_char = [self.pad_chars(i) for i in embeds_char]
        embeds_char = torch.cat(embeds_char, dim=0)
        embeds_char = embeds_char.view(len(embeds_char), 1, -1)
        return embeds_char

    def char_hidden_2_word_net(self, char_hidden):
        hidden = self.char_hidden_2_word(torch.cat(char_hidden))
        word_hidden = self.char_2_word_relu(hidden)
        word_hidden = torch.split(word_hidden, 1)
        return word_hidden

    def forward(self, sentence, chars):
        embeds_char = self.prepare_embed_char(chars)
        lstm_char, char_hidden = self.lstm_char(embeds_char, self.char_hidden)

        word_hidden = self.char_hidden_2_word_net(char_hidden)

        embeds_word = self.word_embeddings(sentence)
        embeds_word = embeds_word.view(len(sentence), 1, -1)
        char_word = torch.cat([embeds_char, embeds_word], dim=2)

        lstm_word, word_hidden = self.lstm_word(char_word, word_hidden)
        lstm_word = pack_padded_sequence(lstm_word, [self.max_words])

        tag_space = self.word_2_tag(lstm_word.data)
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


max_words = max(map(len, sentences))
max_chars = 3
word_embedding_dim = 8
word_hidden_dim = 7
char_embedding_dim = 5
char_hidden_dim = 6

vocab_size = len(word_to_ix)
tagset_size = len(tag_to_ix)

model = LSTMTagger(vocab_size, word_embedding_dim, char_embedding_dim, max_words, max_chars,
                   word_hidden_dim, char_hidden_dim, tagset_size)

loss_fn = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(300):
    total_loss = 0
    for sentence, tag in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        char_in = [prepare_sequence(se.lower(), char_to_ix) for se in sentence]
        targets = prepare_sequence(tag, tag_to_ix)
        tag_scores = model(sentence_in, char_in)
        loss = loss_fn(tag_scores, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0]
    print(total_loss)

se = training_data[1][0]
tag = training_data[1][1]
inputs = prepare_sequence(se, word_to_ix)
char_in = [prepare_sequence(se.lower(), char_to_ix) for se in se]
scores = model(inputs, char_in)
targets = prepare_sequence(tag, tag_to_ix)
torch.max(torch.exp(scores), dim=1)
