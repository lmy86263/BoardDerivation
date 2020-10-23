import numpy as np
from ..LSE import lse


class RidgeRegression(lse.LinearRegression):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha

    def fit(self, X, y):
        fake_inv = np.linalg.inv(X.T @ X + self.alpha * np.eye(X.shape[1])) @ X.T
        self.weight = fake_inv @ y

    def predict(self, X):
        return super(RidgeRegression, self).predict(X)


if __name__ == '__main__':
    pass
