import numpy as np


class LinearRegression:
    def __init__(self):
        self.weight = None

    def fit(self, X, y):
        fake_inv = np.linalg.inv(X.T @ X) @ X.T
        self.weight = fake_inv @ y

    def predict(self, X):
        return self.weight.T @ X


if __name__ == '__main__':
    pass
