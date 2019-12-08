import json

import numpy as np
from sklearn.manifold import TSNE

import datagen
from constants import UNIQ_INDUSTRIES

uniqinds = [x.lower().replace(' ', '_') for x in UNIQ_INDUSTRIES]

sie = datagen._read_startup_industries_tsv('dataset/startup_industries.tsv')
print(len(sie))
names = [name for name, vec in sie.items()]
assert len(set(names) - set(uniqinds)) == 0
assert len(set(uniqinds) - set(names)) == 0

vecs = [vec for name, vec in sie.items()]
vecs = np.array(vecs)

rnd = np.random.RandomState(31372717)

X_embedded = TSNE(n_components=1, random_state=rnd, init='pca', metric='cosine').fit_transform(vecs)
X_embedded = np.squeeze(X_embedded)

d = {i: x for i, x in enumerate(X_embedded)}
sd = sorted(d.items(), key=lambda x: x[1])
print(sd)
for idx, (i, v) in enumerate(sd):
    print(idx, i, names[i], uniqinds.index(names[i]))

print(json.dumps([uniqinds.index(names[i]) for i, v in sd]))
