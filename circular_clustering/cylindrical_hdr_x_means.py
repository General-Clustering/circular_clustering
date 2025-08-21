import numpy as np
from circular_clustering.cylindrical_k_means import CylindricalKMeans
from circular_clustering.cylindrical_hdr import CylindricalHDR

class CylindricalXMeansHDR:
    def __init__(self, X, kmax=20, confidence=0.99, random_state=None):
        self.X = np.asarray(X)
        self.KMax = kmax
        self.confidence = confidence
        self.random_state = random_state

    def fit(self):
        k = self.KMax
        X = self.X

        if X.shape[0] == 1:
            self.labels = np.array([0])
            self.k = 1
            self.centroids = X.copy()
            self.cluster_points = [np.array([True])]
            return self

        stop = False

        while not stop and k >= 1:
            # Apply Cylindrical KMeans
            kmeans = CylindricalKMeans(n_clusters=k, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

            overlap = False
            for i in range(k):
                for j in range(i + 1, k):
                    cluster_i = X[labels == i]
                    cluster_j = X[labels == j]

                    if cluster_i.shape[0] == 0 or cluster_j.shape[0] == 0:
                        continue

                    # Check HDR region overlap
                    if CylindricalHDR.check_hdrs_intersect(cluster_i, cluster_j, alpha=1 - self.confidence):
                        k -= 1
                        overlap = True
                        break
                if overlap:
                    break

            if not overlap:
                stop = True

        # Final clustering
        final_kmeans = CylindricalKMeans(n_clusters=k, random_state=self.random_state)
        self.labels = final_kmeans.fit_predict(X)
        self.k = k
        self.centroids = final_kmeans.cluster_centers_

        # Collect which points belong to which cluster
        self.cluster_points = []
        for cluster_id in range(k):
            mask = self.labels == cluster_id
            self.cluster_points.append(mask)

