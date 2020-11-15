import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from dimensionality_reduction.pca import PCA as PCA_
from contextlib import contextmanager
import time

solvers_ = {"sklearn-PCA": [], "svd-X": [],
            "decompose_var": [], "decompose_T": []}


@contextmanager
def timeblock(n_feature, n_sample, label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        solvers_[label].append([n_feature, n_sample, end - start])


def test_performance():
    n_features = [5, 10, 20, 50, 100]
    n_samples = [50, 100, 200, 300, 500]
    n_components = 2

    def generate_samples(n_feature, n_sample):
        X = []
        for i in range(n_feature):
            x = np.random.randn(n_sample)
            X.append(x)
        X = np.array(X).T
        return X

    fig = plt.figure()
    ax = Axes3D(fig)
    for i, j in [(i, j) for i in n_features for j in n_samples]:
        X = generate_samples(i, j)

        with timeblock(i, j, label="sklearn-PCA"):
            pca = PCA(n_components=n_components)
            pca.fit_transform(X)
        with timeblock(i, j, label="svd-X"):
            svd_pca_ = PCA_(n_features=n_components, solver='svd')
            svd_pca_.reduce(X)
        with timeblock(i, j, label="decompose_var"):
            var_pca_ = PCA_(n_features=n_components, solver='decompose_var')
            var_pca_.reduce(X)
        with timeblock(i, j, label="decompose_T"):
            t_pca_ = PCA_(n_features=n_components, solver='decompose_T')
            t_pca_.reduce(X)

    for label, data in solvers_.items():
        x = np.array(data)[:, 0]
        y = np.array(data)[:, 1]
        z = np.array(data)[:, 2]
        ax.plot3D(x, y, z, label=label)
        ax.scatter3D(x, y, z, label=label)
        ax.legend()

    plt.show()


def test_pca():
    """
            The tree implementations are consistent with sklearn-PCA, if you find the coordinates are different,
            because the scaling factor is unnormalized or the mirror image is not reverted which is normal.
            Orthogonal base can be any one with different scaling factor and same direction.
        """
    fig = plt.figure()
    X = np.array([[-1, -1, 1], [-2, -1, 4], [-3, -2, 1], [1, 1, 6], [2, 1, 5], [3, 2, 7]])

    pca = PCA(n_components=2)
    X_new = pca.fit_transform(X)
    ax1 = fig.add_subplot(221)
    ax1.set_title('Sklearn-PCA')
    ax1.scatter(X_new[:, 0], X_new[:, 1])

    svd_pca_ = PCA_(n_features=2, solver='svd')
    svd_reduced = svd_pca_.reduce(X)
    ax2 = fig.add_subplot(222)
    ax2.set_title('SVD-X')
    ax2.scatter(svd_reduced[:, 0], svd_reduced[:, 1])

    var_pca_ = PCA_(n_features=2, solver='decompose_var')
    var_reduced = var_pca_.reduce(X)
    ax3 = fig.add_subplot(223)
    ax3.set_title('Decompose-variance')
    ax3.scatter(var_reduced[:, 0], var_reduced[:, 1])

    t_pca_ = PCA_(n_features=2, solver='decompose_T')
    t_reduced = t_pca_.reduce(X)
    ax4 = fig.add_subplot(224)
    ax4.set_title('Decompose-T')
    ax4.scatter(t_reduced[:, 0], t_reduced[:, 1])

    plt.show()


if __name__ == '__main__':
    test_performance()
