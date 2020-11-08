import numpy as np


class MarkovChain:
    def __init__(self, transition_matrix, states: list):
        self.transition_matrix = transition_matrix
        self.states = states
        self.state2index = {states[index]: index for index in range(len(states))}
        self.index2state = {index: states[index] for index in range(len(states))}

    def next_state(self, init_state):
        """
            generate next state according to `init_state`
        """
        return np.random.choice(self.states, replace=True,
                                p=self.transition_matrix[self.state2index[init_state]])

    def generate_sequence(self, init_state, steps, end_state=None):
        sequences = [init_state]
        current_state = init_state
        # specify the init_state from state space, mean the initial probability distribution is:
        # p(init_state) = 1, p(others) = 0
        init_prob = 1
        generate_prob = init_prob
        for _ in range(steps):
            next_state = self.next_state(current_state)
            start = self.state2index[current_state]
            end = self.state2index[next_state]
            generate_prob = generate_prob * self.transition_matrix[start][end]
            sequences.append(next_state)
            if next_state == end_state:
                break
            current_state = next_state
        return sequences, generate_prob


if __name__ == '__main__':
    pass
