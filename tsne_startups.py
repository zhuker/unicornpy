import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_vectors.tsv", 'r') as f:
    vectors = np.array(
        [np.array(list(map(lambda x: float(x), line.strip().split('\t'))), dtype=np.float32) for line in f.readlines()])

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_meta.tsv", 'r') as f:
    names = [line.strip() for line in f.readlines()]

if os.path.exists('startups2_emb5.npy'):
    X_embedded = np.load('startups2_emb5.npy')
else:
    rnd = np.random.RandomState(31372717)

    X_embedded = TSNE(n_components=2, perplexity=53, random_state=rnd, init='pca', metric='cosine', n_iter=800,
                      verbose=True).fit_transform(vectors)
    np.save('startups2_emb5.npy', X_embedded)

print(X_embedded.shape)
plt.plot(X_embedded[:, 0], X_embedded[:, 1], '.')
plt.show()

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps=2, min_samples=10).fit(X_embedded)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)

# #############################################################################
# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X_embedded[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col))

    xy = X_embedded[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple(col))

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
