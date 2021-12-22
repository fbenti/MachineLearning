# import os
# os.system('cls' if os.name == 'nt' else 'clear')
from sklearn.cluster import KMeans
import numpy as np

X = np.array((1.0, 1.2, 1.8, 2.3, 2.6, 3.4, 4.0, 4.1, 4.2, 4.6)).reshape(-1,1)

init_cluster = np.array((1.8,3.3,3.6)).reshape(-1,1)

k = 3

kmeans = KMeans(n_clusters=k, init=init_cluster, n_init = 1).fit(X)
cluster = [[] for i in range(k)]
for i in range(len(X)):
    cluster[kmeans.labels_[i]].append(X[i][0])

for i in range(k):
    # c = np.asarray(cluster[i])
    print("\n--- Cluster {} : {} \t mean : {} ---".format(i+1, cluster[i], np.mean(np.asarray(cluster[i]))))