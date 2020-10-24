import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge as Ridge_SK
from ridge import RidgeRegression

if __name__ == '__main__':
    diabetes = load_diabetes()
    # Use only one feature
    diabetes_X = diabetes.data[:, np.newaxis, 2]
    diabetes_y = diabetes.target

    # Split the data into training/testing sets
    X_train, X_test, y_train, y_test = train_test_split(diabetes_X, diabetes_y, test_size=0.3)

    lr_sk = Ridge_SK()
    lr_sk.fit(X_train, y_train)
    y_pred_sk = lr_sk.predict(X_test)

    lr = RidgeRegression()
    lr.fit(X_train, y_train)
    y_pred = lr_sk.predict(X_test)

    plt.scatter(X_test, y_test, color='blue')
    plt.plot(X_test, y_pred_sk, color='red', linewidth=3, label='sklearn', ls='dashed')
    plt.plot(X_test, y_pred, color='green', linewidth=1, label='handwritten', ls='-')

    plt.legend(loc='upper left')
    plt.show()
