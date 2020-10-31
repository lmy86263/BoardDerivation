import numpy as np
from pynverse import inversefunc


class InverseTransformSampling:
    def __init__(self):
        pass

    def sample(self, cdf, n_samples=1000):
        u = np.random.rand(n_samples)
        x = inversefunc(cdf, y_values=u)
        return x



