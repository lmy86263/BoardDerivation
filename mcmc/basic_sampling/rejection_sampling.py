import numpy as np


class RejectionSampling:
    def __init__(self, lower, upper, batch_size=50, proposal_factor=1):
        self.batch_size = batch_size
        self.lower = lower
        self.upper = upper
        self.proposal_factor = proposal_factor

    def sample(self, p, n_samples=1000):
        sampled_results = np.array([])
        while sampled_results.shape[0] < n_samples:
            # use uniform distribution as proposal distribution
            x = np.random.uniform(self.lower, self.upper, self.batch_size)
            y = np.random.rand(self.batch_size)

            batch_sample_results = self.proposal_factor * y < p(x)
            sampled_results = np.concatenate([sampled_results, x[batch_sample_results]])

        return sampled_results[0:n_samples]


if __name__ == '__main__':
    pass
