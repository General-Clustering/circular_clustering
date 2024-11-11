import numpy as np
import scipy

from circular_clustering.kmeans_1d import KMeans1d
from circular_clustering.circular_interval import CircularInterval
from circular_clustering.optimal_circular_kmeans import CircluarKMeans1dOptimal
from circular_clustering.circular_k_means_weighted import CircularWeightedKMeans



class CircularXMeansQuantiles:


    def __init__(self, X, kmax = 20, confidence=0.99, use_optimal_k_means=True):
        self.X = X
        self.num = np.size(self.X, axis=0)
        self.dim = 1
        self.KMax = kmax
        self.confidence = confidence

        if use_optimal_k_means == True:
            self.circular_k_means = CircluarKMeans1dOptimal
        else:
            self.circular_k_means = CircularWeightedKMeans

    def probability_same_cluster(self,mu, sigma, a, b):
        
        return np.abs(scipy.stats.vonmises(loc=mu, kappa=sigma).cdf(b) - scipy.stats.vonmises(loc=mu, kappa=sigma).cdf(a))

    def get_quantiles(self, confidence, kappa, loc):

        a = scipy.stats.vonmises.ppf(1-confidence, kappa=kappa, loc=loc)
        b = scipy.stats.vonmises.ppf(confidence, kappa=kappa, loc=loc)
        return a, b

    def fit(self):
        k = self.KMax
        X = self.X

        while(1):
            ok = k

            #Improve Params
            kmeans = self.circular_k_means(n_clusters=k)
            kmeans.fit(X)
            X=X
            
            labels = kmeans.labels
            addk=0
            
            stop=False
            i=0

            while stop==False and i<=k:
                cli = X[labels == i]
                for j in range(k):
                    if j != i and j>i:    
                        clj = X[labels == j]
                
                        kappai, loci, n = scipy.stats.vonmises.fit(cli)
                        kappaj, locj, n = scipy.stats.vonmises.fit(clj)
            

                        ai, bi = self.get_quantiles(self.confidence, kappai, loci)
                        aj, bj = self.get_quantiles(self.confidence, kappaj, locj)

                        circular_interval_i = CircularInterval(ai, bi, gamma=loci)
                        circular_interval_j = CircularInterval(aj, bj, gamma=locj)
                        

                        overlap = circular_interval_i.intervals_intersect(circular_interval_j)
                        
                        if overlap:
                            addk -= 1
                            stop=True
                            break
                i=i+1
            k += addk


            if ok == k or k <1:
                break


        #Calculate labels and centroids
        kmeans = self.circular_k_means(n_clusters=k)
        kmeans.fit(X)
        self.labels = kmeans.labels
        self.k = k
        self.m = kmeans.cluster_centers_
        self.centroids = kmeans.centroids
        self.cluster_points = kmeans.cluster_points



 