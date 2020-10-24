import numpy as np


class LogisticRegression:
    def __init__(self, max_iteration=1e5, learning_rate=0.001):
        self.weights = None
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1], 1)
        for _ in range(int(self.max_iteration)):
            # update
            sigmoid_y = sigmoid(X @ self.weights)
            error_y = y.reshape(y.shape[0], 1) - sigmoid_y
            gradient = X.T @ error_y
            self.weights -= self.learning_rate * gradient

    def predict(self, X):
        return sigmoid(self.weights.T @ X)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


if __name__ == '__main__':
    pass
