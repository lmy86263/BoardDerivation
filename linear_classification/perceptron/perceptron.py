import numpy as np


class Perceptron:
    def __init__(self, max_iteration=100, learning_rate=0.001):
        self.weight = None
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.weight = np.random.randn(X.shape[0], 1)
        # train with all examples
        for _ in range(self.max_iteration):
            y_pred = self.predict(X)
            mask = y_pred <= 0
            error_count = np.count_nonzero(mask)
            if error_count == 0:
                break
            else:
                # update
                self.weight += self.learning_rate * (X[mask] @ y[mask].T)

    def predict(self, X):
        return np.sign(self.weight.T @ X)


if __name__ == '__main__':
    pass
