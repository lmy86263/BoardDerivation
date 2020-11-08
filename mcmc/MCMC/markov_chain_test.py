from MCMC.markov_chain import MarkovChain
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    states = ['sleep', 'coding', 'derivation']
    transition_matrix = [[0.1, 0.5, 0.4], [0.6, 0.05, 0.35], [0.2, 0.7, 0.1]]
    mc = MarkovChain(transition_matrix, states=states)

    n_steps = 20000
    sequence, _ = mc.generate_sequence(init_state='sleep', steps=n_steps)
    sequence_index = np.array([states.index(value) for value in sequence])

    fig, ax = plt.subplots()

    offsets = range(1, n_steps, 5)
    for i, label in enumerate(states):
        ax.plot(offsets, [np.sum(sequence_index[:offset] == i) / offset for offset in offsets], label=label)
    ax.set_xlabel("number of steps")
    ax.set_ylabel("target distribution")
    ax.legend(frameon=False)
    plt.show()
