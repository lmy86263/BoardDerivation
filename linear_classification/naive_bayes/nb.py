import numpy as np
from collections import Counter
from itertools import product


def gaussian_prob(x, mean, variance):
    constant = 1 / np.sqrt(2 * np.pi * variance)
    exp_part = np.exp((-1 / (2 * variance)) * ((x - mean) ** 2))
    return constant * exp_part


def log_prob(x):
    return np.log(x)


class NaiveBayes:
    def __init__(self):
        self.y_prior = None
        self.y_class = []
        self.y_count = None
        self.dist = {}

    def fit(self, X, y):
        self.y_count = Counter(y)
        self.y_class = np.unique(y)

        self.y_prior = log_prob([count / y.shape[0] for cls, count in self.y_count.items()])

        for pair in list(product(range(X.shape[1]), self.y_class)):
            x_given_y = X[:, pair[0]][y == pair[1]]
            mean = np.around(np.mean(x_given_y), decimals=5)
            var = np.around(np.var(x_given_y), decimals=5)
            self.dist[pair] = [mean, var]

    def predict(self, X):
        likelihood = []
        for cls in self.y_class:
            # given cls, find the likelihood
            example_prob = 0
            for idx in range(X.shape[1]):
                [mean, var] = self.dist[(idx, cls)]
                feature_idx_prob = gaussian_prob(X[:, idx], mean, var)
                feature_idx_log_prob = log_prob(feature_idx_prob)
                example_prob = example_prob + feature_idx_log_prob
            likelihood.append(example_prob)

        total_likliehood = np.array(likelihood).T
        y_post = self.y_prior + total_likliehood

        y_index = np.argmax(y_post, axis=1)
        y_pred = self.y_class[y_index]
        return y_pred


if __name__ == '__main__':
    pass
