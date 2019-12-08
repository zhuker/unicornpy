import torch
from torch.nn import CosineSimilarity, PairwiseDistance

import datagen

sie = datagen._read_startup_industries_tsv('dataset/startup_industries.tsv')
print(len(sie))
cs = CosineSimilarity(dim=0)
# cs = PairwiseDistance(keepdim=True)
_sie = {name: torch.tensor(sie[name]) for name, vec in sie.items()}
s = cs(_sie['medical'], _sie['medical_device'])
print(s)

names = [name for name, v in _sie.items()]
vecs = [v for name, v in _sie.items()]
tt = torch.cat(vecs).view((len(_sie), 100))
# tt = torch.tensor(l)
print(tt.shape)
mv = tt[352]
# mv = tt[2] #torch.mean(tt, 0)
# mv = torch.median(tt, keepdim=True, dim=0)
# mv = mv[0].squeeze()
dists = {i: v.dot(mv).item() for i, v in enumerate(tt)}
sdists = sorted(dists.items(), key=lambda x: x[1])
print(sdists[0], names[sdists[0][0]])
print(sdists[-1], names[sdists[-1][0]])
for idx, dist in sdists:
    print(idx, dist, names[idx])

# -0.572042346882676 352 esports 621 drm
mindist = 100500.
for i, v in enumerate(tt):
    for j, vv in enumerate(tt):
        if i == j:
            continue
        d = v.dot(vv).item()
        # d = cs(v, vv).item()
        if d < mindist:
            print(d, i, names[i], j, names[j])
            mindist = d
