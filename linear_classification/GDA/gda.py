import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal


def multi_gaussian(X, mean, cov):
    var = multivariate_normal(mean, cov)
    prob = var.pdf(X)
    return prob


def log_prob(x):
    return np.log(x)


class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.y_class = None
        self.y_count = None
        self.y_prior = None
        self.means = []
        self.cov = None

    def fit(self, X, y):
        # find prior
        self.y_class, self.y_count = np.unique(y, return_counts=True)
        self.y_prior = log_prob(self.y_count / y.shape[0])
        # find likelihood
        means = []
        examples_var = []
        for cls in self.y_class:
            cls_index = (y == cls).ravel()
            mean = X[cls_index].mean(axis=0)
            means.append(mean)

            var = (X[cls_index] - mean).T @ (X[cls_index] - mean)
            examples_var.append(var)
        self.means = np.array(means)
        sum_of_var = np.sum(examples_var, axis=0)
        self.cov = sum_of_var / y.shape[0]
        pass

    def predict(self, X):
        likelihoods = []
        for cls in self.y_class:
            # given cls, find the likelihood
            likelihood = multi_gaussian(X, self.means[cls], self.cov)
            log_likelihood = log_prob(likelihood)
            likelihoods.append(log_likelihood)

        total_likelihood = np.array(likelihoods).T
        y_post = self.y_prior + total_likelihood

        y_index = np.argmax(y_post, axis=1)
        y_pred = self.y_class[y_index]
        return y_pred


if __name__ == '__main__':
    pass
