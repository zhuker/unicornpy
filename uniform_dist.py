import numpy as np
from sklearn.preprocessing import quantile_transform, StandardScaler, QuantileTransformer

from dataset import read_uir, fastratings

funds, startups, ratings,_,_ = read_uir("dataset4/useritemrating_15.json")
fast = fastratings(ratings)
mlist = [m for _, m in fast.items()]
mlist.append(int(0))
mlist = sorted(mlist)
print(sorted(set(mlist)))
money = np.array(mlist).reshape(-1, 1)

ss = StandardScaler().fit(money)
tmoney = ss.transform(money)
print(money[42], tmoney[42], ss.inverse_transform(tmoney[42]))

qt = QuantileTransformer().fit(money)
qmoney = qt.transform(money)
assert len(qmoney) == len(money)
print(qmoney.min(), qmoney.max())
print(money[24200], qmoney[24200], qt.inverse_transform(qmoney[24200].reshape(-1,1)))
