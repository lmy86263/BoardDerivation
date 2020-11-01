import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from basic_sampling.importance_sampling import ImportanceSampling


def p_x(x, mean, var):
    w1 = 0.3
    w2 = 1 - w1
    g1 = stats.norm(mean[0], var[0]).pdf(x)
    g2 = stats.norm(mean[1], var[1]).pdf(x)
    y = w1 * g1 + w2 * g2
    return y


if __name__ == '__main__':
    mean = [10, 25]
    var = [3, 5]
    # x = np.arange(1, 40, 0.1)
    #
    # y_list = p_x(x, mean, var)
    # plt.plot(x, y_list, c='red')
    #
    # center = (1 + 40) / 2
    # std_var = (center - 1) / 3
    #
    # y_list1 = stats.norm(center, std_var).pdf(x)
    # plt.plot(x, y_list1, c='blue')

    n_samples = [1000, 5000, 10000, 30000, 50000, 100000, 200000, 500000]

    lower = mean[0] - 3 * var[0]
    upper = mean[1] + 3 * var[1]
    i_sampler = ImportanceSampling(lower=lower, upper=upper)

    sample_iteration = 10
    for n_sample in n_samples:
        samples_results = []
        # with the increase of n_samples, the true value has come out gradually
        # the variance also drop down
        for _ in range(sample_iteration):
            expected_value = i_sampler.sample(lambda x: p_x(x, mean, var), n_samples=n_sample)
            samples_results.append(expected_value)
        x = np.ones((len(samples_results))) * n_sample
        plt.scatter(x, samples_results)

    plt.show()
