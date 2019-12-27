import os

import numpy as np

from sklearn.cluster import KMeans

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_vectors.tsv", 'r') as f:
    vectors = np.array(
        [np.array(list(map(lambda x: float(x), line.strip().split('\t'))), dtype=np.float32) for line in f.readlines()])

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_meta.tsv", 'r') as f:
    names = [line.strip() for line in f.readlines()]

if os.path.exists('kmeans_labels.npy'):
    labels = np.load('kmeans_labels.npy')
else:
    db = KMeans(n_clusters=512).fit(vectors)
    labels = db.labels_
    np.save('kmeans_labels.npy', labels)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

C = {}
print(len(labels))
for i, cluster in enumerate(labels):
    name = names[i]
    if cluster not in C:
        C[cluster] = []
    C[cluster].append(name)

s = sorted(C.items(), key=lambda x: len(x[1]), reverse=True)
for k, names in s:
    print(k, len(names), names)
