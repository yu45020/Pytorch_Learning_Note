import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


torch.manual_seed(1)
context_size = 2
embedding_dim = 10
test_sentence = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()


vocab = set(test_sentence)
word_to_ix = {word:i for i, word in enumerate(vocab)}
id_to_word = {i:word for word, i in word_to_ix.items()}
data = []
for i in range(2, len(test_sentence)-2):
    context = [test_sentence[i-2], test_sentence[i-1],
               test_sentence[i+1], test_sentence[i+2]]
    target = test_sentence[i]
    data.append((context, target))


class ContinueBagWord(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).sum(dim=0, keepdim=True)  # sum over words
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_prob = F.log_softmax(out, dim=1)
        return log_prob

loss_func = nn.NLLLoss()
model = ContinueBagWord(len(vocab), embedding_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)

losses = []
for epoch in range(20):
    total_loss = torch.Tensor([0])
    for contex, target in data:
        context_idxs = [word_to_ix[w] for w in contex]
        contex_var = autograd.Variable(torch.LongTensor(context_idxs))
        model.zero_grad()
        log_probs = model(contex_var)
        y_true = autograd.Variable(torch.LongTensor([word_to_ix[target]]))
        loss = loss_func(log_probs, y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    print(total_loss[0])
    losses.append(total_loss[0])

import matplotlib.pyplot as plt

plt.plot(losses)

""" processes are abstract beings that inhabit """
contest_test = ["processes","spells", "that", 'conjure']
contest_test_idxs = [word_to_ix[w] for w in contest_test]
contest_test_var = autograd.Variable(torch.LongTensor(contest_test_idxs))
probs = model.forward(contest_test_var)
a,b = torch.max(probs, 1)
id_to_word[b.data[0]]





word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(3, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([0,1])
hello_embed = embeds(autograd.Variable(lookup_tensor))
hello_embed.sum(dim=0)
print(hello_embed)