import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn import datasets
from bayesian_lse import BayesianLinearRegression
from sklearn.linear_model import LinearRegression as LinearRegression
from sklearn import metrics
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_regression


def random_regression_problem():

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # X, y = datasets.load_boston(return_X_y=True)
    # X = X[:, np.newaxis, 5]
    X, y = make_regression(
        n_samples=1000,
        n_features=1,
        n_targets=1,
        noise=50
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # identify outliers in the training dataset
    iso = IsolationForest()
    y_hat = iso.fit_predict(X_test)
    mask = y_hat != -1
    X_test_hat, y_test_hat = X_test[mask, :], y_test[mask]

    index = 1
    fig = plt.figure()

    for i, j, label in [(X_test, y_test, 'Not Remove outliers'), (X_test_hat, y_test_hat, 'Remove outliers')]:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(i)

        bayesian_lr = BayesianLinearRegression(n_features=X.shape[1])
        bayesian_lr.fit(X_train, y_train)
        y_pred_bayes, _ = bayesian_lr.predict(i)

        # when outliers exist, Bayesian LR is more sensitive
        # when no outliers exist, MAP = LSE = MLE with noise ~ N(0, sigma**2)
        print(label, "Linear regression error: ", metrics.mean_absolute_error(j, y_pred_lr))
        print(label, "Bayesian Regression error", metrics.mean_absolute_error(j, y_pred_bayes))

        ax = fig.add_subplot(120 + index)
        ax.set_title(label)
        ax.scatter(i, j, color='blue', s=1)

        ax.plot(i, y_pred_lr, color='red', linewidth=3, label='sklearn', ls='dashed')
        ax.scatter(i, y_pred_bayes, color='green', linewidth=1, label='Bayesian Linear regression')

        ax.legend(loc='upper left')

        index = index + 1

    plt.show()
