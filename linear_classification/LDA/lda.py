import numpy as np


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.weights = None
        self.center = None

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1], 1)
        self.weights /= np.linalg.norm(self.weights, ord=2)

        mask = y <= 0
        # class
        mu_c1 = np.mean(X[~mask], axis=0)
        mu_c2 = np.mean(X[mask], axis=0)
        sigma_c1 = np.cov(X[~mask].T)
        sigma_c2 = np.cov(X[mask].T)
        s_w = sigma_c1 + sigma_c2
        s_w_inv = np.linalg.inv(s_w)
        self.center = (mu_c1 + mu_c2)/2
        self.weights = s_w_inv @ (mu_c1 - mu_c2)

    def predict(self, X):
        y_pred = (X-self.center.T) @ self.weights.reshape(self.weights.shape[0], 1)
        return np.sign(y_pred)


if __name__ == '__main__':
    pass
