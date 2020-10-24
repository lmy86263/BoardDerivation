import numpy as np


class Perceptron:
    def __init__(self, max_iteration=1e5, learning_rate=0.01):
        self.weight = None
        self.max_iteration = max_iteration
        self.learning_rate = learning_rate

    def fit(self, X, y):
        self.weight = np.random.randn(X.shape[1], 1)
        # train with all examples
        for _ in range(int(self.max_iteration)):
            y_pred = self.predict(X)
            mask = y_pred <= 0
            error_count = np.count_nonzero(mask)
            if error_count == 0:
                break
            else:
                # update
                a = X[mask].T
                b = y.reshape(y.shape[0], 1)[mask]
                gradient = - a @ b
                self.weight -= self.learning_rate * gradient

    def predict(self, X):
        return np.sign(X @ self.weight).ravel()


if __name__ == '__main__':
    pass
