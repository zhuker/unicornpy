import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.cluster import DBSCAN, OPTICS
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_vectors.tsv", 'r') as f:
    vectors = np.array(
        [np.array(list(map(lambda x: float(x), line.strip().split('\t'))), dtype=np.float32) for line in f.readlines()])

with open("/home/zhukov/clients/unicorn/unicorn/dataset4/startups_emb5_meta.tsv", 'r') as f:
    names = [line.strip() for line in f.readlines()]

# if os.path.exists('startups2_emb5.npy'):
#     X_embedded = np.load('startups2_emb5.npy')
# else:
#     rnd = np.random.RandomState(31372717)
#
#     X_embedded = TSNE(n_components=2, perplexity=53, random_state=rnd, init='                                             pca', metric='cosine', n_iter=800,
#                       verbose=True).fit_transform(vectors)
#     np.save('startups2_emb5.npy', X_embedded)


# #############################################################################
# Compute DBSCAN
db = OPTICS(min_samples=10).fit(vectors)
# db = DBSCAN(eps=0.3, min_samples=10).fit(vectors)
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

