import numpy as np
import random
from scipy.stats import vonmises

def circ_dist(point, data_pts, r):
        min_dists = []
        for data in data_pts:
            min_dist = np.min([np.abs(point - data), r - np.abs(point - data)])
            min_dists.append(min_dist)
        return min_dists

class CircularWeightedKMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.iteration = 0
        self.centroids = None
        self.prev_centroids = None
        self.std = None
        self.r = None
        self.fac = None
        self.clust_weights = None
        self.cluster_points = []
        self.labels = []
        
    def fit(self, X_train, weights=None, r=2*np.pi):
        self.r = r
        self.fac = 2*np.pi/self.r
        self.centroids = [random.choice(X_train)]
        weights = np.ones(len(X_train))
        for _ in range(self.n_clusters-1):
            dists = circ_dist(self.centroids[-1], X_train, self.r)
            dists /= np.sum(dists)
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        while np.not_equal(self.centroids, self.prev_centroids).any() and self.iteration < self.max_iter:
            self.prev_centroids = self.centroids
            self.clust_weights = [0]*self.n_clusters
            self.std = [0]*self.n_clusters
            self.cluster_points = [[] for _ in range(self.n_clusters)]
            sorted_weights = [[] for _ in range(self.n_clusters)]
            for x, w in zip(X_train, weights):
                # assign each point to closest centroid
                dists = circ_dist(x, self.centroids, self.r)
                closest_centroid_idx = np.argmin(dists)
                self.clust_weights[closest_centroid_idx] += w
                self.cluster_points[closest_centroid_idx].append(x)
                sorted_weights[closest_centroid_idx].append(w)
            for idx, _ in enumerate(self.centroids):
                # weighted mean of all points in cluster gives new centroid
                self.std[idx], self.centroids[idx], n = vonmises.fit(np.array(self.cluster_points[idx]))

                self.clust_weights[idx] = np.sum(np.array(sorted_weights[idx]))
            self.clust_weights = [w/np.sum(self.clust_weights) for w in self.clust_weights]
            self.iteration += 1
            self.cluster_centers_ = len(self.centroids)
            self.get_labels(X_train)

    def get_labels(self,X_train): 
        label_list = list(range(len(self.cluster_points)))
        self.labels = np.zeros(len(X_train))
        for i in range(len(self.cluster_points)):
            cl = self.cluster_points[i] 
            label = label_list[i]
            for x in cl:
                index = np.where(X_train==x)[0][0]
                self.labels[index] = label
                
    
    def plot(self, X_train, weights):
        # Histogram 
        plt.hist(X_train, bins=20, weights=weights)
        sizes = self.clust_weights/np.max(self.clust_weights)*200
        # draw std as lines 
        for i, centroid in enumerate(self.centroids):
            # random color
            color = np.random.rand(3,)
            plt.scatter(centroid, max(weights)/2*sizes[i]/max(sizes), s=sizes[i], zorder=2, color=color, label=f'Cluster {i}, x={int(centroid)}, weight {self.clust_weights[i]/np.sum(self.clust_weights):.2f}, std {self.std[i]:.2f}')
            y_range = 2*[max(weights)/2*sizes[i]/max(sizes)]
            plt.plot([centroid-self.std[i], centroid+self.std[i]], y_range, zorder=1, color=color)
        plt.legend()
        plt.show()