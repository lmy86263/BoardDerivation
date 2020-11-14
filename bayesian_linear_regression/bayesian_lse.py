import numpy as np
from scipy import stats


class BayesianLinearRegression:
    def __init__(self, n_features, noise_mean=0, noise_var=0.5, alpha=0.5):
        # learning parameters
        self.weight_cov = None
        self.weight_mean = None
        # noise dist, noise ~ N(noise, noise_var)
        self.noise_mean = noise_mean
        self.noise_var = noise_var
        # the default prior is designed for computation easily because of conjugate distribution for Gaussian
        self.weight_prior_mean = 0
        self.weight_prior_var = np.eye(n_features) / alpha

    def fit(self, X, y):
        precision_matrix = (1 / self.noise_var) * X.T @ X + np.linalg.inv(self.weight_prior_var)
        self.weight_cov = np.linalg.inv(precision_matrix)
        self.weight_mean = (1 / self.noise_var) * self.weight_cov @ X.T @ y
        return self

    def predict(self, X):
        predicted_mean = X @ self.weight_mean
        predicted_cov = X @ self.weight_cov @ X.T + self.noise_var

        return predicted_mean, \
            [stats.norm(loc=predicted_mean, scale=predicted_cov[i][i] ** 0.5) for i in range(X.shape[0])]


if __name__ == '__main__':
    pass
