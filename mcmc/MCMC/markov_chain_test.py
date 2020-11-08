from MCMC.markov_chain import MarkovChain
import matplotlib.pyplot as plt

if __name__ == '__main__':
    states = ['sleep', 'coding', 'derivation']
    transition_matrix = [[0.1, 0.5, 0.4], [0.6, 0.05, 0.35], [0.2, 0.7, 0.1]]
    mc = MarkovChain(transition_matrix, states=states)

    all_sequences = []
    steps = [1000, 5000, 10000, 20000, 50000, 70000, 100000]
    for i in steps:
        sequence, _ = mc.generate_sequence(init_state='sleep', steps=1000)
        all_sequences.append(sequence)

    labels = [str(step) for step in steps]
    plt.hist(all_sequences, histtype='bar', label=labels)
    plt.legend()
    plt.show()
