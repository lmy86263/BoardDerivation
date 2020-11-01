import numpy as np
from scipy import stats


class ImportanceSampling:
    """
        The 'sampling' is not what it suggests, actually. It is used for
        approximation of expected value.
    """
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.center = None
        self.std_var = None
        pass

    def gaussian_importance(self, n_samples):
        self.center = (self.lower + self.upper)/2
        self.std_var = (self.center - self.lower)/3
        samples = np.random.normal(self.center, self.std_var, n_samples)
        return samples

    def sample(self, g, n_samples):
        samples = self.gaussian_importance(n_samples)

        f = stats.norm(self.center, self.std_var).pdf(samples)
        weights = g(samples) / f

        expected_value = (samples @ weights)/samples.shape[0]
        return expected_value


if __name__ == '__main__':
    pass
