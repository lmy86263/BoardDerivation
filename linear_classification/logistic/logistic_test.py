from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.linear_model import LogisticRegression as LogisticRegression_SK

import matplotlib.pyplot as plt
import numpy as np

from logistic import LogisticRegression


def generate_classification_examples():
    X, y = make_classification(n_samples=1000, n_classes=2,
                               n_features=2, n_clusters_per_class=1,
                               n_informative=2, n_redundant=0)
    return X, y


if __name__ == '__main__':
    X, y = generate_classification_examples()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
    x1_examples = np.linspace(np.min(X) - 0.05 * (np.max(X) - np.min(X)),
                              np.max(X) + 0.05 * (np.max(X) - np.min(X)),
                              100)

    logistic_sk = LogisticRegression_SK()
    logistic_sk.fit(X, y)
    sk_weights = logistic_sk.coef_.reshape(logistic_sk.n_features_in_, -1)
    x2_examples_SK = (sk_weights[0] * x1_examples) / sk_weights[1]

    logistic = LogisticRegression()
    logistic.fit(X, y)
    x2_examples = (logistic.weights[0] * x1_examples) / logistic.weights[1]

    plt.plot(x1_examples, x2_examples, c='red', label='board derivation')
    plt.plot(x1_examples, x2_examples_SK, c='green', label='Sklearn')
    plt.legend()
    plt.show()
