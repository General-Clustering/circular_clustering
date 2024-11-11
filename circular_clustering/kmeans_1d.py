import numpy as np
from ckmeans_1d_dp import ckmeans

class KMeans1d:


    def __init__(self, n_clusters):
        self.n_clusters= n_clusters

    def fit(self, X):
        self.labels_ = ckmeans(X, k=self.n_clusters).cluster
        self.cluster_points = self._organize_in_cluster_points(X, self.labels_,self.n_clusters)
        self.cluster_centers_ = []
        for cl in self.cluster_points:
            self.cluster_centers_.append(cl.mean())
        
    
    def _organize_in_cluster_points(self, X, clusters_indexes, k):
        cluster_points = []
        for i in range(0,k):
            indexes = self._all_indices(i,clusters_indexes)
            cl = np.array([X[j] for j in indexes])
            cluster_points.append(cl)
        return cluster_points
    
    def _all_indices(self, value, qlist):
        qlist = list(qlist)
        indices = []
        idx = -1
        while True:
            try:
                idx = qlist.index(value, idx+1)
                indices.append(idx)
            except ValueError:
                break
        return indices