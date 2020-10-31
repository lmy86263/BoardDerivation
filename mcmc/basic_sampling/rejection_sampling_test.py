from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from basic_sampling.rejection_sampling import RejectionSampling


def pdf(x):
    return np.exp(-x)


if __name__ == '__main__':
    lower = 0
    upper = 4
    x = np.arange(lower, upper, 0.01)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    rs = RejectionSampling(lower=lower, upper=upper)
    x_sample = rs.sample(pdf, n_samples=50000)
    ax1.hist(x_sample, bins=100, facecolor="green", edgecolor="black", alpha=0.7)
    ax1.set_ylabel('samples')

    ax2 = ax1.twinx()
    ax2.plot(x, pdf(x), 'r')
    ax2.set_ylabel('Y values for exp(-x)')

    plt.show()
