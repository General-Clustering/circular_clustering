import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Rely on your existing implementations
from circular_clustering.cylindrical_hdr import CylindricalHDR

TAU = 2.0 * np.pi


# ==============================
# Angle helpers
# ==============================

def wrap_pi(theta: np.ndarray) -> np.ndarray:
    # Wrap angles to (-pi, pi]
    return (theta + np.pi) % TAU - np.pi

def ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Minimal signed difference a - b in (-pi, pi]
    return wrap_pi(a - b)


# ==============================
# Cylindrical geometry helpers
# ==============================

def embed_cylinder(theta: np.ndarray, z: np.ndarray, z_scale: float) -> np.ndarray:
    # Embed (theta, z) into R^3 as [cos theta, sin theta, z / z_scale].
    # Squared Euclidean distance in this embedding approximates:
    # chordal angular distance^2 + (delta z / z_scale)^2.
    return np.column_stack([np.cos(theta), np.sin(theta), z / z_scale])

def auto_z_scale(z: np.ndarray, eps: float = 1e-8) -> float:
    # Choose a global scale for z so its typical variation matches the unit circle embedding.
    s = float(np.std(z))
    return s + eps

def pairwise_sq_dists_embedded(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Squared Euclidean distance between rows of A and B (fast).
    A2 = np.sum(A * A, axis=1, keepdims=True)
    B2 = np.sum(B * B, axis=1, keepdims=True).T
    D2 = A2 + B2 - 2.0 * (A @ B.T)
    return np.maximum(D2, 0.0)


# ==============================
# k-means++ seeding on cylinder
# ==============================

class _CylindricalKPP:
    # Minimal k-means++ initializer on a cylinder for data (theta, z).
    # - Embed to R^3 via [cos(theta), sin(theta), z/z_scale].
    # - Choose k seeds with D^2 sampling.
    # - Assign each point to nearest seed (Voronoi) once.

    def __init__(self, n_clusters: int, z_scale: float, random_state: Optional[int] = None):
        self.k = int(n_clusters)
        self.z_scale = float(z_scale)
        self.random_state = random_state
        self.centers_: Optional[np.ndarray] = None   # centers in original (theta,z)
        self.labels_: Optional[np.ndarray] = None

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _init_plus_plus(self, X: np.ndarray) -> np.ndarray:
        # Return up to k initial centers chosen by k-means++ from rows of X (theta,z).
        n = X.shape[0]
        rng = self._rng()
        # embed
        E = embed_cylinder(X[:, 0], X[:, 1], self.z_scale)

        # first center uniformly
        first_idx = rng.integers(0, n)
        centers_idx = [first_idx]
        centers_emb = [E[first_idx]]

        # remaining with D^2
        for _ in range(1, self.k):
            D2 = np.min(pairwise_sq_dists_embedded(E, np.vstack(centers_emb)), axis=1)
            D2 = np.maximum(D2, 0.0)
            total = float(np.sum(D2))
            if not np.isfinite(total) or total <= 0.0:
                break
            p = D2 / total
            p = p / float(np.sum(p))  # strict renormalization
            nxt = rng.choice(n, p=p)
            centers_idx.append(nxt)
            centers_emb.append(E[nxt])

        return X[np.array(centers_idx, dtype=int)]  # (k0, 2) in (theta,z)

    def fit(self, X: np.ndarray) -> "_CylindricalKPP":
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == 2, "X must be (n,2) with columns (theta,z)"
        X = X.copy()
        X[:, 0] = wrap_pi(X[:, 0])

        centers = self._init_plus_plus(X)              # (k0,2)
        E = embed_cylinder(X[:, 0], X[:, 1], self.z_scale)
        Ce = embed_cylinder(centers[:, 0], centers[:, 1], self.z_scale)
        D2 = pairwise_sq_dists_embedded(E, Ce)         # (n,k0)
        labels = np.argmin(D2, axis=1)

        self.centers_ = centers
        self.labels_ = labels
        return self


# ==============================
# Cylindrical k++ HDR merge
# ==============================

class CylindricalKMeansPPHDRMerge:
    # Start with kmax k-means++ seeds on cylindrical data X[:,(theta,z)], assign once,
    # then merge clusters whose cylindrical HDR regions intersect; repeat until stable.
    #
    # Exposes back-compat fields after fit():
    #   - labels, k, m, centroids, cluster_points
    # Also sklearn-style:
    #   - labels_, n_clusters_, cluster_centers_

    def __init__(
        self,
        X: np.ndarray,
        kmax: int = 20,
        confidence: float = 0.99,
        random_state: Optional[int] = None,
        z_scale: Optional[float] = None,
        max_iter: int = 100,
        *,
        init: str = "k-means++",                 # NEW: "k-means++" | "manual" | "random"
        init_centers: Optional[np.ndarray] = None,   # NEW: shape (k0, 2) [(theta,z)]
        init_indices: Optional[np.ndarray] = None,   # NEW: integer indices into X
    ) -> None:
        X = np.asarray(X, dtype=float)
        assert X.ndim == 2 and X.shape[1] == 2, "X must be (n,2) with columns (theta,z)"
        self.X = X.copy()
        self.X[:, 0] = wrap_pi(self.X[:, 0])

        self.KMax = int(kmax)
        self.confidence = float(confidence)
        self.random_state = random_state
        self.max_iter = int(max_iter)

        # scaling of z for k++ distances
        self.z_scale = float(z_scale) if z_scale is not None else auto_z_scale(self.X[:, 1])

        # initialization options
        self.init = init
        self.init_centers = init_centers
        self.init_indices = init_indices

        # outputs
        self.labels_: Optional[np.ndarray] = None
        self.n_clusters_: Optional[int] = None
        self.cluster_centers_: Optional[np.ndarray] = None  # (k,2) (theta_mean, z_mean)

        # back-compat
        self.labels = None
        self.k = None
        self.m = None
        self.centroids = None
        self.cluster_points: Optional[List[np.ndarray]] = None

    @staticmethod
    def _connected_components(adj: np.ndarray) -> List[np.ndarray]:
        k = adj.shape[0]
        seen = np.zeros(k, dtype=bool)
        comps: List[List[int]] = []
        for s in range(k):
            if seen[s]:
                continue
            stack = [s]
            seen[s] = True
            comp = [s]
            while stack:
                v = stack.pop()
                for u in np.flatnonzero(adj[v]):
                    if not seen[u]:
                        seen[u] = True
                        stack.append(u)
                        comp.append(u)
            comps.append(np.array(sorted(comp), dtype=int))
        return comps

    def _assign_to_centers(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        # helper: assign rows of X to nearest 'centers' in the embedded metric
        E = embed_cylinder(X[:, 0], X[:, 1], self.z_scale)
        Ce = embed_cylinder(centers[:, 0], centers[:, 1], self.z_scale)
        D2 = pairwise_sq_dists_embedded(E, Ce)
        return np.argmin(D2, axis=1)

    def _manual_or_random_init(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.random_state)
        if self.init == "manual":
            if self.init_indices is not None:
                idx = np.asarray(self.init_indices, dtype=int).ravel()
                if idx.size == 0:
                    raise ValueError("init_indices is empty.")
                centers = X[idx]
            elif self.init_centers is not None:
                centers = np.asarray(self.init_centers, dtype=float)
                if centers.ndim != 2 or centers.shape[1] != 2:
                    raise ValueError("init_centers must have shape (k0, 2) for (theta, z).")
                centers = centers.copy()
                centers[:, 0] = wrap_pi(centers[:, 0])  # wrap theta
            else:
                raise ValueError("init='manual' requires init_indices or init_centers.")
        else:  # "random"
            k0 = min(self.KMax, X.shape[0])
            idx = rng.choice(X.shape[0], size=k0, replace=False)
            centers = X[idx]
        labels = self._assign_to_centers(X, centers)
        return centers, labels

    def fit(self) -> "CylindricalKMeansPPHDRMerge":
        X = self.X
        n = X.shape[0]

        # 1) Seeding + one Voronoi assignment
        if self.init in ("manual", "random"):
            centers0, labels0 = self._manual_or_random_init(X)
        else:  # "k-means++" (default)
            kpp = _CylindricalKPP(n_clusters=self.KMax, z_scale=self.z_scale, random_state=self.random_state).fit(X)
            labels0 = kpp.labels_
            centers0 = kpp.centers_

        # Initial clusters (drop empty)
        clusters: List[np.ndarray] = []
        for j in range(centers0.shape[0]):
            idxs = np.flatnonzero(labels0 == j)
            if idxs.size > 0:
                clusters.append(idxs)

        # 2) iterative HDR-based merging
        it = 0
        alpha = 1.0 - self.confidence
        while it < self.max_iter:
            it += 1
            k = len(clusters)
            if k <= 1:
                break

            adj = np.zeros((k, k), dtype=bool)
            any_overlap = False
            for i in range(k):
                Xi = X[clusters[i]]
                for j in range(i + 1, k):
                    Xj = X[clusters[j]]
                    try:
                        inter = CylindricalHDR.check_hdrs_intersect(Xi, Xj, alpha=alpha)
                    except Exception:
                        # conservative: if HDR check fails, merge
                        inter = True
                    if inter:
                        adj[i, j] = adj[j, i] = True
                        any_overlap = True

            if not any_overlap:
                break

            comps = self._connected_components(adj)
            new_clusters: List[np.ndarray] = []
            for comp in comps:
                merged = np.unique(np.concatenate([clusters[c] for c in comp], axis=0))
                new_clusters.append(merged)
            clusters = new_clusters

        # 3) finalize: labels_, centers, etc.
        labels = np.empty(n, dtype=int)
        for cid, idxs in enumerate(clusters):
            labels[idxs] = cid
        self.labels_ = labels
        self.n_clusters_ = len(clusters)

        # cluster centers: circular mean over theta, arithmetic mean over z
        centers = np.zeros((self.n_clusters_, 2), dtype=float)
        for cid, idxs in enumerate(clusters):
            theta_i = X[idxs, 0]
            z_i = X[idxs, 1]
            c = np.cos(theta_i).mean()
            s = np.sin(theta_i).mean()
            mu_theta = float(np.arctan2(s, c))
            mu_z = float(np.mean(z_i))
            centers[cid, :] = [mu_theta, mu_z]
        self.cluster_centers_ = centers

        # back-compat aliases & cluster point indices
        self.labels = self.labels_
        self.k = self.n_clusters_
        self.m = self.n_clusters_
        self.centroids = self.cluster_centers_
        self.cluster_points = clusters

        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        # Assign new points to nearest final center using the same embedded metric.
        if self.cluster_centers_ is None:
            raise RuntimeError("Call fit() before predict().")
        X_new = np.asarray(X_new, dtype=float)
        assert X_new.ndim == 2 and X_new.shape[1] == 2, "X_new must be (n,2)"
        X_new = X_new.copy()
        X_new[:, 0] = wrap_pi(X_new[:, 0])

        # embed
        E_new = embed_cylinder(X_new[:, 0], X_new[:, 1], self.z_scale)
        E_c = embed_cylinder(self.cluster_centers_[:, 0], self.cluster_centers_[:, 1], self.z_scale)
        D2 = pairwise_sq_dists_embedded(E_new, E_c)
        return np.argmin(D2, axis=1)

