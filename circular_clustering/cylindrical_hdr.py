import numpy as np
from numpy.linalg import inv, LinAlgError, eigh

class CylindricalHDR:
    def __init__(self, y_margin=1.0, grid_size=600):
        self.y_margin = y_margin

    def fit(self, X):
        """
        X is expected to be (n, 2) with columns (theta, y),
        where theta in [-pi, pi]
        """
        theta = X[:, 0]
        y = X[:, 1]
        x1 = np.cos(theta)
        x2 = np.sin(theta)
        x3 = y
        self.X_3d = np.column_stack([x1, x2, x3])

        self.mean_ = np.mean(self.X_3d, axis=0)
        self.cov_ = np.cov(self.X_3d, rowvar=False)

        # Try inverting covariance matrix, add small ridge if singular
        try:
            self.inv_cov_ = inv(self.cov_)
        except LinAlgError:
            print("⚠️ Covariance matrix singular, adding ridge regularization and retrying inversion...")
            ridge = 1e-8 * np.eye(self.cov_.shape[0])
            try:
                self.inv_cov_ = inv(self.cov_ + ridge)
            except LinAlgError:
                print("⚠️ Still singular after regularization, using pseudo-inverse instead.")
                self.inv_cov_ = np.linalg.pinv(self.cov_ + ridge)

    def get_hdr_threshold(self, alpha):
        """
        Computes the squared Mahalanobis distance threshold for the given HDR level.
        This assumes the ellipsoidal distribution on the cylinder.
        """
        from scipy.stats.distributions import chi2
        return chi2.ppf(1-alpha, df=3)

    def inside_hdr(self, X, alpha):
        """
        Returns a boolean mask of which points (in original [theta, y] space)
        lie inside the HDR region.
        """
        theta = X[:, 0]
        y = X[:, 1]
        X_embedded = np.column_stack([np.cos(theta), np.sin(theta), y])
        diffs = X_embedded - self.mean_
        dists = np.einsum("ij,jk,ik->i", diffs, self.inv_cov_, diffs)
        return dists <= self.get_hdr_threshold(alpha)

    def get_hdr_region(self, alpha):
        """
        Returns a function that checks if a point is inside the HDR region.
        Useful for intersection testing.
        """
        threshold = self.get_hdr_threshold(alpha)
        mu = self.mean_
        inv_cov = self.inv_cov_

        def inside(x):
            # x is [theta, y]
            c = np.cos(x[0])
            s = np.sin(x[0])
            y = x[1]
            diff = np.array([c, s, y]) - mu
            d = diff @ inv_cov @ diff
            return d <= threshold

        return inside

    @staticmethod
    def check_hdrs_intersect(X1, X2, alpha=0.9, samples=10000):
        """
        Returns True if the HDR regions of X1 and X2 intersect, based on
        sampling from both clusters and checking overlap in HDR membership.
        """
        hdr1 = CylindricalHDR()
        hdr2 = CylindricalHDR()
        hdr1.fit(X1)
        hdr2.fit(X2)

        in1 = hdr1.get_hdr_region(alpha)
        in2 = hdr2.get_hdr_region(alpha)

        # Sample from both clusters and check mutual HDR inclusion
        combined = np.vstack([
            X1[np.random.choice(len(X1), size=min(samples, len(X1)), replace=True)],
            X2[np.random.choice(len(X2), size=min(samples, len(X2)), replace=True)]
        ])

        count = 0
        for x in combined:
            if in1(x) and in2(x):
                count += 1
                break

        return count > 0

    def plot_hdr_only(self, alpha=0.9, ax=None, color="red", label=None):
        """
        Plots the HDR region as a 2D scatter of points within the HDR on the cylinder.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        theta = np.linspace(-np.pi, np.pi, self.grid_size if hasattr(self, 'grid_size') else 600)
        y = np.linspace(
            self.X_3d[:, 2].min() - self.y_margin,
            self.X_3d[:, 2].max() + self.y_margin,
            self.grid_size if hasattr(self, 'grid_size') else 600
        )
        Theta, Y = np.meshgrid(theta, y)
        points = np.column_stack([Theta.ravel(), Y.ravel()])
        mask = self.inside_hdr(points, alpha)
        ax.contourf(
            Theta, Y, mask.reshape(Theta.shape),
            levels=[0.5, 1], alpha=0.3, colors=[color], label=label
        )
        if label:
            ax.plot([], [], color=color, label=label)  # Legend entry

        ax.set_xlabel("θ (radians)")
        ax.set_ylabel("y")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_title("HDR Region")

        return ax

