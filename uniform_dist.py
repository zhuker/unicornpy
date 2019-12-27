import numpy as np
from sklearn.preprocessing import quantile_transform, StandardScaler

from dataset import read_uir, fastratings

funds, startups, ratings = read_uir("dataset4/useritemrating_15.json")
fast = fastratings(ratings)
mlist = [m for _, m in fast.items()]
mlist.append(int(0))
mlist = sorted(mlist)
print(sorted(set(mlist)))
money = np.array(mlist).reshape(-1, 1)
qt = quantile_transform(money, n_quantiles=1000, random_state=0, copy=True)
print(qt)
ss = StandardScaler().fit_transform(money)
print(ss)
