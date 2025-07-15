# circular_clustering
Adaptation of X means algorithm for circular data

Install the package using:

```bash
pip install circular-clustering
```

## X means algorithm with quantiles

The class `CircularXMeansQuantiles` contains the X means algorithm for circular data. The use is similar to 
the clustering algorithms in `scipy`.

To import it:

```python
from circular_clustering.circular_x_means_quantiles import CircularXMeansQuantiles
```

To invoke the class:

```python
circXmeans = CircularXMeansQuantiles(x, kmax=8, confidence=0.99, use_optimal_k_means=True)
```

- **x must be a one-dimensional NumPy array of angles between `-π` and `π`.**
- `kmax=` sets the maximum number of clusters.

To fit the algorithm:

```python
circXmeans.fit()
```

Centroids are available at `circXmeans.centroids`, and labels at `circXmeans.labels`.

### Example (circular data):

```python
import numpy as np
import matplotlib.pyplot as plt

from circular_clustering.circular_x_means_quantiles import CircularXMeansQuantiles

x = np.array([ 1.658,  1.369,  1.783,  1.587,  0.942,  1.268,
               1.740,  2.245,  1.955,  1.132, -1.694, -1.121,
              -1.249, -1.834, -1.868, -1.351, -1.492, -1.607,
              -1.323, -1.913,  0.099,  0.060, -0.074, -0.127,
               0.179,  0.006,  0.273, -0.285,  0.080,  0.301])

circXmeans = CircularXMeansQuantiles(x, kmax=8, confidence=0.99, use_optimal_k_means=True)
circXmeans.fit()

plt.figure(figsize=(5,5))
plt.axes().set_aspect('equal', 'datalim')
plt.scatter(np.cos(x), np.sin(x))

for c in circXmeans.centroids:
    plt.scatter(np.cos(c), np.sin(c), c="r")

for cl in circXmeans.cluster_points:
    plt.scatter(np.cos(cl), np.sin(cl), c=np.random.rand(3,))

plt.show()
```

---

## Cylindrical clustering with HDR-based XMeans

The `XMeansHDR` class supports clustering in cylindrical coordinates, where data has both an angular and linear component (θ, y). Clustering is performed using HDR-based region separation and a custom cylindrical distance metric.

To import:

```python
from circular_clustering.cylindrical_hdr_x_means import XMeansHDR
```

### Example (cylindrical data):

```python
import numpy as np
import matplotlib.pyplot as plt
from circular_clustering import XMeansHDR

def make_cluster(center_theta, center_y, spread_theta, spread_y, n):
    angles = np.random.vonmises(center_theta, 1 / (spread_theta ** 2), size=n)
    heights = np.random.normal(center_y, spread_y, size=n)
    return np.column_stack([angles, heights])

# Simulation data
np.random.seed(123)
true_k = 4
points_per_cluster = 200
spread_theta = 0.25
spread_y = 0.4
alpha = 0.1
confidence = 1 - alpha

centers_theta = np.random.uniform(-np.pi, np.pi, true_k)
centers_y = np.random.uniform(-3, 3, true_k)

clusters = [
    make_cluster(c_theta, c_y, spread_theta, spread_y, points_per_cluster)
    for c_theta, c_y in zip(centers_theta, centers_y)
]
X = np.vstack(clusters)

# Fit HDR XMeans
xmeans = XMeansHDR(X, kmax=6, confidence=confidence)
xmeans.fit()

# Plot result
colors = plt.cm.tab10.colors
plt.figure(figsize=(8, 6))
for i in range(xmeans.k):
    cluster_points = X[xmeans.labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i % 10], alpha=0.6, label=f"Cluster {i}")

plt.xlabel("Angle θ (radians)")
plt.ylabel("Height y")
plt.title(f"Cylindrical XMeans Clustering\nTrue: {true_k}, Found: {xmeans.k}")
plt.grid(True)
plt.legend()
plt.show()
```

