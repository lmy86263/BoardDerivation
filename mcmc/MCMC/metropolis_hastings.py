import numpy as np
from scipy import stats


class MetropolisHastings:
    def __init__(self):
        pass

    def gaussian_transition_kernel(self, current_state):
        new_state = np.random.normal(current_state, 1)
        return new_state

    def gaussian_transition_kernel_prob(self, current_state, new_state):
        prob = stats.norm(current_state, 1).pdf(new_state)
        return prob

    def cauchy_transition_kernel(self, current_state):
        new_state = stats.cauchy.rvs(loc=current_state)
        return new_state

    def cauchy_transition_kernel_prob(self, current_state, new_state):
        prob = stats.cauchy.pdf(new_state, loc=current_state)
        return prob

    def sample(self, target_dist, n_samples=1000, burn_in_samples=500, interval=5, kernel='gaussian'):
        """
        :param kernel:
        :param interval: transition kernel from which new state is drawn
        :param target_dist:
        :param n_samples:
        :param burn_in_samples: considering the samples in burn-in period which is not in stationary distribution
        :return:
        """
        accepted_sequence = []
        # random init a state
        current_state = np.random.normal(0, 1)
        transition_kernel_prob = None
        transition_kernel = None
        if kernel == 'gaussian':
            transition_kernel = self.gaussian_transition_kernel
            transition_kernel_prob = self.gaussian_transition_kernel_prob
        elif kernel == 'cauchy':
            transition_kernel = self.cauchy_transition_kernel
            transition_kernel_prob = self.cauchy_transition_kernel_prob

        for _ in range(interval * n_samples + burn_in_samples):
            u = np.random.rand()
            new_state = transition_kernel(current_state)
            accepted_state = new_state

            # if target distribution is not normalized, the MH sampling is still available, because the normalized
            # factor is removed when computing the accepted ratio
            new_state_part = target_dist(new_state) * transition_kernel_prob(new_state, current_state)
            current_state_part = target_dist(current_state) * transition_kernel_prob(current_state, new_state)

            accepted_ratio = min(1, new_state_part / current_state_part)
            if u >= accepted_ratio:
                accepted_state = current_state
            accepted_sequence.append(accepted_state)
            current_state = accepted_state
        candidate_sequence = accepted_sequence[burn_in_samples:]
        revised_sequence = [state for i, state in enumerate(candidate_sequence) if i % interval == 0]
        return revised_sequence


if __name__ == '__main__':
    pass
