import numpy as np

import lasagne.init as LI
from lasagne.random import get_rng

class SpectralRadius(LI.Initializer):
    """
    Rescale weights to given spectral radius. Based on echo state network approach.
    See e.g. Herbert Jaeger. The “echo state” approach to analysing and training recurrent neural networks.
    Technical Report GMD Report 148, German National Research Center for Information Technology, 2001.
    """
    def __init__(self, initializer=LI.Uniform(0.5), radius=1.25, density=1.0):
        self.initializer = initializer
        self.radius = radius
        self.density = density

    def sample(self, shape):
        W = self.initializer.sample(shape)
        if self.density < 1.0:
            N = np.prod(W.shape)
            drop_ix = get_rng().choice(N, size=int((1.0-self.density)*N), replace=False)
            W.reshape(-1,)[drop_ix] = 0.0
        lmax = np.max(np.abs(np.linalg.eig(W)[0]))
        return self.radius*W/lmax