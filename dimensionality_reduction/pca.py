import numpy as np


class PCA:
    def __init__(self, n_features=2, solver='auto'):
        self.n_features = n_features
        self.singular_values = None
        self.singular_vectors = None
        self.solver = solver
        self.solvers = {"svd": self.svd_x,
                        "decompose_var": self.eigen_decomposition_var,
                        "decompose_T": self.eigen_decomposition_T}

    def svd_x(self, x, centering_matrix):
        """ decompose the samples: X after centering """
        decompose_target = centering_matrix @ x
        u, s, v_t = np.linalg.svd(decompose_target)

        sorted_indices = np.argsort([i ** 2 for i in s])
        self.singular_values = s[sorted_indices[:-self.n_features - 1:-1]]
        self.singular_vectors = v_t.T
        self.singular_vectors = self.singular_vectors[:, sorted_indices[:-self.n_features - 1:-1]]

        normalized_factor = np.linalg.norm(self.singular_vectors, axis=0)
        self.singular_vectors = self.singular_vectors / normalized_factor

        x_reduced = x @ self.singular_vectors
        return x_reduced

    def eigen_decomposition_var(self, x, centering_matrix):
        """ decompose the samples' variance """
        variance = (x.T @ centering_matrix @ x) / x.shape[0]
        eigen_values, eigen_vectors = np.linalg.eig(variance)
        sorted_indices = np.argsort(eigen_values)
        self.singular_values = eigen_values[sorted_indices[:-self.n_features - 1:-1]]
        self.singular_vectors = eigen_vectors[:, sorted_indices[:-self.n_features - 1:-1]]

        normalized_factor = np.linalg.norm(self.singular_vectors, axis=0)
        self.singular_vectors = self.singular_vectors / normalized_factor

        x_reduced = x @ self.singular_vectors
        return x_reduced

    def eigen_decomposition_T(self, x, centering_matrix):
        """ decompose the T, its advantages is more fast when the #(dimensions) >> #(samples), like in images,
            and it generates coordinates directly, called `Principle Coordinate Analysis`
        """
        part = centering_matrix @ x
        decompose_target = part @ part.T
        eigen_values, eigen_vectors = np.linalg.eig(decompose_target)

        sorted_indices = np.argsort(eigen_values)
        self.singular_values = eigen_values[sorted_indices[:-self.n_features - 1:-1]]
        self.singular_vectors = eigen_vectors[:, sorted_indices[:-self.n_features - 1:-1]]

        normalized_factor = np.linalg.norm(self.singular_vectors, axis=0)
        self.singular_vectors = self.singular_vectors / normalized_factor

        x_reduced = self.singular_vectors
        return x_reduced

    def reduce(self, x):
        samples_n = x.shape[0]
        centering_matrix = np.eye(samples_n) - (1 / samples_n) * (np.ones(samples_n).reshape(samples_n, -1)
                                                                  @ np.ones(samples_n).T.reshape(-1, samples_n))
        x_reduced = None
        if self.solver == 'auto':
            samples_number = x.shape[0]
            dimension = x.shape[1]
            if samples_number > dimension:
                x_reduced = self.svd_x(x, centering_matrix)
            else:
                x_reduced = self.eigen_decomposition_T(x, centering_matrix)
        else:
            x_reduced = self.solvers[self.solver](x, centering_matrix)

        return x_reduced


if __name__ == '__main__':
    pass
