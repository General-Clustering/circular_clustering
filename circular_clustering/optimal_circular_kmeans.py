from scipy.stats import vonmises
import numpy as np
from ckmeans_1d_dp import ckmeans

class CircluarKMeans1dOptimal:


    def __init__(self, n_clusters):
        self.n_clusters= n_clusters

    def fit(self, X):
        labels_iter = []
        cluster_points_iter = []
        cluster_centers_iter = []
        distances_cluster_iter = []
        moved_data_iter = []
        errors = []

        for x in X:
            data = self.move_data(X, x)
            moved_data_iter.append(data)
            labels_ = ckmeans(data, k=self.n_clusters).cluster
            cluster_points = []
            cluster_points_moved = []
            for i in range(self.n_clusters):
                data = np.array(data)
                cluster_points_moved.append(data[labels_==i])
                cluster_points.append(X[labels_==i])
            cluster_centers_ = []
            errors_cl =[]
            
            dists_coll = []
            for i in range(len(cluster_points_moved)):
                cl =cluster_points[i]
                cl_moved =cluster_points_moved[i]
                std, mean_vm, n = vonmises.fit(cl)
                #mean = mean%(2*np.pi)
                #z_mean = np.sin(mean) + 1j*np.cos(mean)
                #cl_z = np.sin(cl) + 1j*np.cos(cl)
                #distances_cluster = np.abs([np.angle(z- z_mean) for z in cl_z])
                
                mean = cl_moved.mean()
                distances_cluster = np.abs(cl_moved-mean)**2

                cluster_centers_.append(mean_vm)
                dists_coll.append(distances_cluster)
                error = distances_cluster.mean()
                errors_cl.append(error)

            distances_cluster_iter.append(dists_coll) 
            errors_cl = np.array(errors_cl)
            sum_err = errors_cl.sum()
            errors.append(sum_err)     
            labels_iter.append(labels_)
            cluster_points_iter.append(cluster_points)
            cluster_centers_iter.append(cluster_centers_)

        index_min = np.argmin(errors, axis=-1)
        self.moved_data = moved_data_iter[index_min]
        self.moved_data_iter = moved_data_iter
        self.cluster_points = cluster_points_iter[index_min]
        self.cluster_points_iter = cluster_points_iter
        self.cluster_centers_ = cluster_centers_iter[index_min]
        self.cluster_centers_iter = cluster_centers_iter
        self.centroids = cluster_centers_iter[index_min]
        self.labels = labels_iter[index_min]
        self.labels_iter = labels_iter
        self.errors = errors
        self.distances_cluster_iter = distances_cluster_iter
        
    def move_data(self, X, x):
        data = ((X  - x))% (2 * np.pi)
        return data
    
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