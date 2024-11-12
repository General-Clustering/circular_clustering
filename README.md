# circular_clustering
Adaptation of X means algorithm for circular data

install the package using

    pip install circular-clustering

# X means algorithm with quantiles

The class CircularXMeansQuantiles contains the X means algorithm for circular data. The use is similar to 
the clustering algorithms in scipy.

To import it use:

    from circular_clustering.circular_x_means_quantiles import CircularXMeansQuantiles

To invoke the class use:

    circXmeans = CircularXMeansQuantiles(x, kmax=8, confidence=0.99, use_optimal_k_means=True)

- Notice that **x will be asumed to be a one dimensional numpy array of floats between `-2*np.pi` and `2*np.pi`**
- Note that `kmax=` will fix the maximun number of clusters desired.

To fit the algorithm:

    circXmeans.fit()

Centroids are available at `circXmeans.centroids`, while labels are available at `circXmeans.labels`

# Example of use

    import numpy as np
    import matplotlib.pyplot as plt

    from circular_clustering.circular_x_means_quantiles import CircularXMeansQuantiles


    x = np.array([ 1.65824313,  1.36928704,  1.78387914,  1.5872978 ,  0.94256866,  1.26848715,
      1.74052966,  2.24549453,  1.95567787,  1.13242718, -1.69442172, -1.1219816 ,
     -1.24971631, -1.8345882 , -1.86868335, -1.35164974, -1.49251909, -1.60791295,
     -1.32335839, -1.91367625,  0.09918781,  0.06024964, -0.07459622, -0.12782416,
      0.17969188,  0.00629938,  0.27335795, -0.28549939,  0.08032582,  0.30140874])


    circXmeans = CircularXMeansQuantiles(x, kmax=8, confidence=0.99, use_optimal_k_means=True)
    circXmeans.fit()
    centroids = circXmeans.centroids

    print("number of clusters found: ", len(circXmeans.cluster_points))


    plt.figure(figsize=(5,5))
    plt.axes().set_aspect('equal', 'datalim')  # before `plt.show()`
    plt.scatter(np.cos(x),np.sin(x))

    for c in centroids:
        plt.scatter(np.cos(c),np.sin(c), c="r")
    
    for i in range(len(circXmeans.cluster_points)):
        cl = circXmeans.cluster_points[i]
        plt.scatter(np.cos(cl),np.sin(cl), c=np.random.rand(3,))
    plt.show()

    circXmeans.labels

![result](./doc/clusters_example.png)

