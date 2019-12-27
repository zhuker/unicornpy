from math import sqrt

import torch
import json
import numpy as np
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from dataset import fastratings, getrating, read_uir


class DenseNet(nn.Module):

    def __init__(self, n_users, n_items, n_factors, H1, D_out):
        """
        Simple Feedforward with Embeddings
        """
        super().__init__()
        # user and item embedding layers
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)
        # linear layers
        self.linear1 = torch.nn.Linear(n_factors * 2, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, users, items):
        users_embedding = self.user_factors(users)
        items_embedding = self.item_factors(items)
        # concatenate user and item embeddings to form input
        x = torch.cat([users_embedding, items_embedding], 1)
        h1_relu = F.relu(self.linear1(x))
        output_scores = self.linear2(h1_relu)
        return F.relu(output_scores)

    def predict(self, users, items):
        # return the score
        output_scores = self.forward(users, items)
        return output_scores


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # create user embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        # create item embeddings
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)

    def forward(self, user, item):
        # matrix multiplication
        return (self.user_factors(user) * self.item_factors(item)).sum(1)

    def predict(self, user, item):
        return self.forward(user, item)

funds, startups, ratings = read_uir("dataset4/useritemrating_15.json")

n_users = len(funds)
n_items = len(startups)

print(len(ratings))
print(n_users)
print(n_items)
print(len(ratings) * 100.0 / (n_users * n_items))

fast = fastratings(ratings)
mlist = [m for _, m in fast.items()]
print(sorted(set(mlist)))
money = np.array(mlist)
print(np.median(money))

min_money = 0
ptp_money = np.ptp(money)

_min = min(money)
_max = max(money)
print(_min, _max)


def money01(m):
    return m / _max


print(money01(_min), money01(_max))

print(getrating(fast, 10, 1))
print(money01(getrating(fast, 10, 1)))

# model = MatrixFactorization(n_users, n_items, n_factors=16)
N = 32
model = DenseNet(n_users, n_items, n_factors=N, H1=N, D_out=1)
loss_fn = torch.nn.MSELoss()
optimizer = SGD(model.parameters(), lr=1e-6)


class RatingsDataset(Dataset):
    def __init__(self, ratingslist):
        self.fast = list(fastratings(ratingslist).items())

    def __getitem__(self, item):
        ui, r = self.fast[item]
        u = ui >> 32
        i = ui & 0xffffffff
        return u, i, money01(r)

    def __len__(self):
        return len(self.fast)


trainset = RatingsDataset(ratings)
trainloader = DataLoader(trainset, shuffle=True, batch_size=256, drop_last=True)

dev = torch.device('cuda')
model.to(dev)

EPOCHS = 20
for epoch in range(0, EPOCHS):
    for i, (user, item, rating) in enumerate(trainloader):
        user = user.to(dev)
        item = item.to(dev)
        rating = rating.to(dev)
        # predict
        prediction = model(user, item).squeeze()
        loss = loss_fn(prediction, rating)
        if i % 100 == 0:
            print(epoch, i, loss.item(), int(sqrt(loss.item()) * _max))
            print((rating[0:8].detach().cpu().numpy() * _max // 1000).astype(np.int))
            print((prediction[0:8].detach().cpu().numpy() * _max // 1000).astype(np.int))

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
