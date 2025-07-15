import numpy as np
from numpy.linalg import norm
from sklearn.base import BaseEstimator, ClusterMixin

class CylindricalKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _cylindrical_distance(self, a, b):
        """
        Compute squared distance between points a and b on the cylinder.
        a, b: (theta, y) with theta in [-pi, pi]
        """
        # Angular distance (wrapped)
        dtheta = np.angle(np.exp(1j * (a[0] - b[0])))  # wrap to [-pi, pi]
        dy = a[1] - b[1]
        return dtheta ** 2 + dy ** 2

    def _init_centers(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centers):
        labels = np.zeros(len(X), dtype=int)
        for i, x in enumerate(X):
            dists = [self._cylindrical_distance(x, c) for c in centers]
            labels[i] = np.argmin(dists)
        return labels

    def _update_centers(self, X, labels):
        centers = []
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) == 0:
                # Random reinitialization if a cluster gets empty
                centers.append(X[np.random.choice(len(X))])
                continue

            # Circular mean for theta
            sin_sum = np.sum(np.sin(cluster_points[:, 0]))
            cos_sum = np.sum(np.cos(cluster_points[:, 0]))
            mean_theta = np.arctan2(sin_sum, cos_sum)

            # Regular mean for y
            mean_y = np.mean(cluster_points[:, 1])
            centers.append([mean_theta, mean_y])
        return np.array(centers)

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.cluster_centers_ = self._init_centers(X)

        for i in range(self.max_iter):
            labels = self._assign_clusters(X, self.cluster_centers_)
            new_centers = self._update_centers(X, labels)

            shift = np.max([self._cylindrical_distance(c1, c2) for c1, c2 in zip(self.cluster_centers_, new_centers)])
            self.cluster_centers_ = new_centers
            if shift < self.tol:
                break

        self.labels_ = self._assign_clusters(X, self.cluster_centers_)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self._assign_clusters(X, self.cluster_centers_)

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

