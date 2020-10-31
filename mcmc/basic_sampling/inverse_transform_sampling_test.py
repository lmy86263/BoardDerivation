import numpy as np
import matplotlib.pyplot as plt
from basic_sampling.inverse_transform_sampling import InverseTransformSampling


def cdf(x):
    return 1 - np.exp(-x)


def pdf(x):
    return np.exp(-x)


if __name__ == '__main__':
    its_generator = InverseTransformSampling()
    samples = its_generator.sample(cdf, n_samples=10000)
    x = np.arange(0.1, 4, 0.0001)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.hist(samples, bins=100, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    ax1.set_ylabel('samples')

    ax2 = ax1.twinx()
    ax2.plot(x, pdf(x), 'r')
    ax2.set_ylabel('Y values for exp(-x)')

    plt.show()
