from sklearn.datasets import make_classification, make_blobs, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.naive_bayes import GaussianNB as GaussianNB

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from nb import NaiveBayes


def generate_examples():
    iris = load_iris()
    X = iris.data[0:100, [0, 2]]
    y = iris.target[0:100]

    # X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
    #                            random_state=1, n_clusters_per_class=1)

    return X, y


def plot_decision_regions(X, y, classifiers: list, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    num_of_plots = len(classifiers)
    f, axs = plt.subplots(1, num_of_plots, sharex='col', sharey='row', figsize=(10, 8))

    for idx, classifier, tt in zip(range(num_of_plots), classifiers, ['sklearn', 'self-handwritten']):
        # plot the decision surface
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = np.array(Z).reshape(xx1.shape)

        axs[idx].contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
        l = axs[idx].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        axs[idx].set_title(tt)
        axs[idx].set_xlabel('petal length [standardized]')
        axs[idx].set_ylabel('petal width [standardized]')


        # plot examples
        for index, cl in enumerate(np.unique(y)):
            axs[idx].scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                        alpha=0.8, c=colors[index],
                        marker=markers[index],
                        label=cl, edgecolor='black')
        # highlight test data
        if test_idx:
            X_test, y_test = X[test_idx, :], y[test_idx]
            axs[idx].scatter(X_test[:, 0], X_test[:, 1],
                        c='', edgecolor='black', alpha=1.0, linewidth=1,
                        marker='o', s=100, label='test set')


if __name__ == '__main__':
    X, y = generate_examples()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    sc.fit(X)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    nb_sk = GaussianNB()
    nb_sk.fit(X_train, y_train)

    nb1 = NaiveBayes()
    nb1.fit(X_train, y_train)
    nb1.predict(X_test[0:3, :])

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined_std = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std,
                          y=y_combined_std,
                          classifiers=[nb_sk, nb1],
                          test_idx=range(70, 100))
    plt.tight_layout()
    plt.show()
