import math
from math import sqrt, nan

import torch
import json
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from dataset import fastratings, getrating, read_uir, read_json


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** math.ceil(math.log2(x))


class DenseNet(nn.Module):

    def __init__(self, n_users, n_items, n_inds, n_ftypes, n_factors, H1, D_out):
        """
        Simple Feedforward with Embeddings
        """
        super().__init__()
        # user and item embedding layers
        self.user_factors = torch.nn.Embedding(n_users, n_factors,
                                               sparse=True)
        self.item_factors = torch.nn.Embedding(n_items, n_factors,
                                               sparse=True)
        ind_sz = 16 #math.ceil(math.log(n_inds)) * 3
        self.industry_factors = torch.nn.Embedding(n_inds, ind_sz,
                                                   sparse=True)
        ftype_sz = 4 #math.ceil(math.log(n_ftypes))
        self.ftype_factors = torch.nn.Embedding(n_ftypes, ftype_sz,
                                                sparse=True)
        # linear layers
        H0 = n_factors + n_factors + ind_sz + ftype_sz
        # H0 = n_factors + n_factors + 0 + ftype_sz
        # H0 = n_factors + n_factors + 0 + 0
        # self.linear0 = torch.nn.Linear(H0, H0)
        self.linear1 = torch.nn.Linear(H0, H0 // 2)
        self.linear2 = torch.nn.Linear(H0 // 2, H0 // 4)
        self.linear3 = torch.nn.Linear(H0 // 4, D_out)
        print(self)

    def forward(self, funds, startups, industries, funding_type):
        nz = (industries != 0)
        inds__sum = nz.sum(dim=1)
        ii = self.industry_factors(industries)
        ii[~nz] = 0
        # print(inds__sum.detach().cpu().numpy())
        ind_avg = ii.sum(dim=1) / inds__sum.unsqueeze(1)
        users_embedding = self.user_factors(funds)
        items_embedding = self.item_factors(startups)
        ftype_embedding = self.ftype_factors(funding_type)
        # concatenate user and item embeddings to form input
        # x = torch.cat([users_embedding, ind_avg], 1)
        # x = torch.cat([users_embedding, items_embedding], 1)
        x = torch.cat([users_embedding, items_embedding, ftype_embedding, ind_avg], 1)
        # x = torch.cat([users_embedding, items_embedding, ftype_embedding], 1)
        out = x
        # out = F.relu(self.linear0(out))
        out = F.relu(self.linear1(out))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        return out

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


useritemrating = read_json("useritemrating_ind_ftype_15.json")
ratings = useritemrating['ratings']
funds = useritemrating['funds']
startups = useritemrating['startups']
startup_industries = useritemrating.get('startupIndustries', {})
industries = useritemrating.get('industries', {})
ftypes = useritemrating.get('ftypes', {})

n_users = len(funds)
n_items = len(startups)

print(len(ratings))
print(n_users)
print(n_items)
print(len(ratings) * 100.0 / (n_users * n_items))

mlist = [m for u, i, m, f in ratings]
mlist.append(int(0))
print(sorted(set(mlist)))
money = np.array(mlist)
print(np.median(money))

#qt = QuantileTransformer().fit(money.reshape(-1, 1))
_max = max(money)

def money01(m: int):
    return m / _max #qt.transform([[m]])[0, 0]


def scale_money(m: np.ndarray) -> np.ndarray:
    #transform = qt.inverse_transform(m.reshape(-1, 1))
    # return np.squeeze(transform)
    return m * _max

# model = MatrixFactorization(n_users, n_items, n_factors=16)
N = 32
model = DenseNet(n_users, n_items, len(industries), len(ftypes), n_factors=N, H1=N, D_out=1)
loss_fn = torch.nn.MSELoss()
optimizer = SGD(model.parameters(), lr=1e-6)


class RatingsDataset(Dataset):
    def __init__(self, ratingslist):
        self.ratingslist = ratingslist  # list(fastratings(ratingslist).items())

    def __getitem__(self, item):
        u, i, r, ft = self.ratingslist[item]
        inds = startup_industries[str(i)]
        assert inds.count(0) == 0, f'zero elements unexpected {inds}'
        z = np.zeros(22, dtype=np.int64)
        z[0:len(inds)] = inds
        assert inds is not None
        return u, i, money01(r), z, ft

    def __len__(self):
        return len(self.ratingslist)


trainset = RatingsDataset(ratings)
trainloader = DataLoader(trainset, shuffle=True, batch_size=256, drop_last=True)

dev = torch.device('cuda')
model.to(dev)

EPOCHS = 100500
for epoch in range(0, EPOCHS):
    for i, (user, item, rating, inds, ft) in enumerate(trainloader):
        optimizer.zero_grad()
        user = user.to(dev)
        item = item.to(dev)
        rating = rating.to(dev)
        inds = inds.to(dev)
        ft = ft.to(dev)
        # predict
        prediction = model(user, item, inds, ft).squeeze()
        loss = loss_fn(prediction, rating)
        if i % 100 == 0:
            loss_item = loss.item()
            print(epoch, i, loss_item, scale_money(np.array([sqrt(loss_item)])))
            assert not math.isnan(loss_item)
            print((scale_money(rating[0:8].detach().cpu().numpy()) // 1000).astype(np.int))
            print((scale_money(prediction[0:8].detach().cpu().numpy()) // 1000).astype(np.int))

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
