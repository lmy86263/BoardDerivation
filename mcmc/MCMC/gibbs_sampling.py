import numpy as np


class GibbsSampling:
    def __init__(self):
        pass

    def sample(self, dimensions, condition_dist, n_samples=1000, burn_in=100, interval=3):
        current_state = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]])
        sampled_sequence = []
        for _ in range(interval * n_samples + burn_in):
            new_state = np.array([None]*dimensions)
            for i in range(dimensions):
                dimension_index = np.ma.array(range(dimensions), mask=True)
                dimension_index.mask[i] = False
                conditions = current_state[dimension_index.mask]

                new_state_elements = new_state[new_state != None]
                length = len(new_state_elements)
                if length > 0:
                    # mean new state element is sampled, need to update the condition
                    conditions = np.hstack((new_state_elements, conditions[length:]))
                new_state_element = condition_dist(conditions, i)
                new_state[i] = new_state_element
            current_state = new_state
            sampled_sequence.append(new_state.astype(np.float64))
        candidate_sequence = sampled_sequence[burn_in:]
        revised_sequence = [state for i, state in enumerate(candidate_sequence) if i % interval == 0]
        return np.array(revised_sequence)


if __name__ == '__main__':
    pass
