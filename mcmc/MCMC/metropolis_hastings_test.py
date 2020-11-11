import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from MCMC.metropolis_hastings import MetropolisHastings
import datetime


def p_x(x, mean, var):
    """
    the target distribution is defined in (-infinity, + infinity), so if the transition kernel is defined in (0,
    +infinity), the sampling will fail, and the samples cannot be good as we expect
    """
    w1 = 0.3
    w2 = 1 - w1
    g1 = stats.norm(mean[0], var[0]).pdf(x)
    g2 = stats.norm(mean[1], var[1]).pdf(x)
    y = w1 * g1 + w2 * g2
    return y


def gamma_dist(x):
    y = stats.gamma.pdf(x, a=1.99)
    return y


def test_burn_in_impact():
    mean = [10, 25]
    var = [3, 5]
    x = np.arange(1, 40, 0.1)

    n_sample = 5000
    mh_sampler = MetropolisHastings()

    fig = plt.figure()
    burn_in_periods = [0, 100, 500, 1000, 2000, 3000]
    for i in range(len(burn_in_periods)):
        ax1 = fig.add_subplot(231 + i)
        burn_in_period = burn_in_periods[i]
        sample_sequence = mh_sampler.sample(lambda x: p_x(x, mean, var), n_samples=n_sample,
                                            burn_in_samples=burn_in_period)
        ax1.hist(sample_sequence, bins=100, facecolor="green", edgecolor="black", alpha=0.7)
        ax1.set_ylabel('samples')
        ax1.set_title('burn-in periods=' + str(burn_in_period))

        ax2 = ax1.twinx()
        ax2.plot(x, p_x(x, mean, var), 'r')
        ax2.set_ylabel('Y values for target distribution')

    plt.show()


def test_interval_impact():
    mean = [10, 25]
    var = [3, 5]
    x = np.arange(1, 40, 0.1)

    n_sample = 5000
    mh_sampler = MetropolisHastings()

    fig = plt.figure()
    intervals = [0, 3, 5, 10, 20, 50]
    burn_in_period = 3000
    for i in range(len(intervals)):
        ax1 = fig.add_subplot(231 + i)
        interval = intervals[i]
        sample_sequence = mh_sampler.sample(lambda x: p_x(x, mean, var), n_samples=n_sample,
                                            burn_in_samples=burn_in_period, interval=interval)
        ax1.hist(sample_sequence, bins=100, facecolor="green", edgecolor="black", alpha=0.7)
        ax1.set_ylabel('samples')
        ax1.set_title('interval=' + str(interval))

        ax2 = ax1.twinx()
        ax2.plot(x, p_x(x, mean, var), 'r')
        ax2.set_ylabel('Y values for target distribution')

    plt.show()


def test_different_kernel():
    x = np.arange(1, 10, 0.001)

    n_sample = 10000
    mh_sampler = MetropolisHastings()

    fig = plt.figure()
    kernels = ['gaussian', 'cauchy']
    burn_in_period = 3000
    interval = 20
    for i in range(len(kernels)):
        ax1 = fig.add_subplot(121 + i)
        kernel = kernels[i]
        start = datetime.datetime.now()
        sample_sequence = mh_sampler.sample(lambda x: gamma_dist(x), n_samples=n_sample,
                                            burn_in_samples=burn_in_period, interval=interval, kernel=kernel)
        end = datetime.datetime.now()
        ax1.hist(sample_sequence, bins=100, facecolor="green", edgecolor="black", alpha=0.7)
        ax1.set_ylabel('samples')
        ax1.set_title('kernel=' + kernel + ", run time=" + str((end - start).seconds))

        ax2 = ax1.twinx()
        ax2.plot(x, gamma_dist(x), 'r')
        ax2.set_ylabel('Y values for target distribution')

    plt.show()


if __name__ == '__main__':
    # test_burn_in_impact()
    # test_interval_impact()
    test_different_kernel()
