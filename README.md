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
![result](./doc/clusters_example.png)


---

## Cylindrical clustering with HDR-based XMeans

The `CylindricalXMeansHDR` class supports clustering in cylindrical coordinates, where data has both an angular and linear component (θ, y). Clustering is performed using HDR-based region separation and a custom cylindrical distance metric.

To import:

```python
from circular_clustering import CylindricalXMeansHDR
```

### Example (cylindrical data):

```python

import numpy as np
import matplotlib.pyplot as plt
from circular_clustering.cylindrical_hdr_x_means import CylindricalXMeansHDR

# Fix seed for reproducibility
np.random.seed(42)

# Function to create elliptical clusters on the cylinder
def make_cluster(center_theta, center_y, spread_theta, spread_y, n=100):
    theta = np.random.vonmises(center_theta, 1 / (spread_theta ** 2), size=n)
    y = np.random.normal(center_y, spread_y, size=n)
    return np.column_stack([theta, y])

# One cluster near -π, one near π, one at 0
X = np.vstack([
    make_cluster(np.pi - 0.2, 0.5, 0.15, 0.2),     # Cluster near +π
    make_cluster(-np.pi + 0.2, -0.5, 0.15, 0.2),   # Cluster near -π (should wrap!)
    make_cluster(0.0, 1.5, 0.2, 0.2),              # Central cluster
])

# Run HDR-based X-Means clustering
alpha = 0.3
xmeans = CylindricalXMeansHDR(X, kmax=6, confidence=1 - alpha, random_state=0)
xmeans.fit()
print(f"Found clusters: {xmeans.k}")

# Plot results
colors = plt.cm.tab10.colors
plt.figure(figsize=(8, 6))
for i in range(xmeans.k):
    cluster = X[xmeans.labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i % 10], alpha=0.6, s=20, label=f"Cluster {i}")

plt.title(f"CylindricalXMeansHDR clustering (wraparound test)\nFound {xmeans.k} clusters")
plt.xlabel("Angle θ (radians)")
plt.ylabel("Height y")
plt.xlim(-np.pi, np.pi)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

![result](./doc/clusters_example2.png)
