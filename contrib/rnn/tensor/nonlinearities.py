import theano.tensor as TT

def logisticmap(x):
    return TT.maximum(1. - x**2, -1.)

class SaturatingLogmap(object):
    ''' saturating logistic map'''
    def __init__(self, alpha):
        self._alpha = alpha
    
    def __call__(self, x):
        return 1. - self._alpha*TT.tanh(x)**2

salm = SaturatingLogmap(1.8)
chaotic_salm = SaturatingLogmap(3.225)