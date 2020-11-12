import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.cm as cm
from scipy import stats
import numpy as np
from MCMC.gibbs_sampling import GibbsSampling


def target_dist(x, mean, cov):
    """
    we use 2-dimension gaussian distribution as target distribution, the condition probability
    is difficult to get the close-form in high dimensions.
    :return:
    """
    prob = stats.multivariate_normal.pdf(x, mean=mean, cov=cov)
    return prob


def target_condition_dist(condition, mean, cov, sample_dim):
    """this part is coding according to bishop: Pattern recognition& Machine learning"""
    all_index = np.ma.array(list(range(len(mean))), mask=True)
    all_index.mask[sample_dim] = False
    conditions_part_mean = mean[all_index.mask]
    sample_mean = mean[sample_dim]

    sample = None
    if len(mean) == 2:
        print("2-dimension gaussian case")
        condition_dim = 1 - sample_dim
        condition_mean = sample_mean + cov[sample_dim][condition_dim] * (1 / cov[condition_dim][condition_dim]) * (
                    condition - conditions_part_mean)
        condition_var = cov[sample_dim][sample_dim] - cov[sample_dim][condition_dim] * (
                    1 / cov[condition_dim][condition_dim]) * cov[condition_dim][sample_dim]
        sample = np.random.normal(condition_mean.astype(np.float64), np.sqrt(condition_var))
    else:
        print("high dimension case")
    return sample[0]


def test():
    x = np.arange(1, 10)
    y = x.reshape(-1, 1)
    h = x * y

    cs = plt.contourf(h, levels=[10, 30, 50],
                      colors=['#808080', '#A0A0A0', '#C0C0C0'], extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.show()


if __name__ == '__main__':
    mean = np.array([1, 2])
    cov = np.array([[2, 0.5], [0.5, 3]])

    # x = np.arange(-5, 7, 0.1)
    # y = np.arange(-6, 11, 0.1)
    # x, y = np.meshgrid(x, y)
    # data = np.dstack([x, y])
    # prob = target_dist(data, mean=mean, cov=cov)
    #
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # # surf = ax.plot_surface(x, y, prob, rstride=1, cstride=1, cmap=cm.coolwarm,
    # #                        linewidth=0, antialiased=False)
    # # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.contourf(x, y, prob)
    init_state = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    gibbs_sampler = GibbsSampling(init_state=init_state)
    samples = gibbs_sampler.sample(dimensions=2, condition_dist=lambda x, y: target_condition_dist(x, mean, cov, y),
                                   n_samples=1000)
    x = samples[:, 0]
    y = samples[:, 1]

    x, y = np.meshgrid(x, y)
    data = np.dstack([x, y])
    prob = target_dist(data, mean=mean, cov=cov)

    fig = plt.figure()
    ax = Axes3D(fig)
    # surf = ax.plot_surface(x, y, prob, rstride=1, cstride=1, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    # fig.colorbar(surf, shrink=0.5, aspect=5)

    # plt.scatter(x, y)
    ax.contourf(x, y, prob)
    plt.show()

