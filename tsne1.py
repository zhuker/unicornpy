import json

import numpy as np
from sklearn.manifold import TSNE

import datagen
from dataset import UnicornDataset

ud = UnicornDataset.from_json('dataset2/startups2.jsonl', 'dataset2/rounds2.jsonl')
uniqinds = [x.lower().replace(' ', '_') for x in ud.uniq_industries]
for i, ind in enumerate(uniqinds):
    print(i, ind)

sie = datagen._read_startup_industries_tsv('dataset2/industry_emb2_t5.tsv')
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
with open('dataset2/industry_sort.json', 'w') as f:
    f.write(json.dumps([uniqinds.index(names[i]) for i, v in sd]))
