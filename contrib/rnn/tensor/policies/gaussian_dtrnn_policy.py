import lasagne.nonlinearities as NL

from contrib.rnn.tensor.policies.base import GaussianRNNPolicy
from contrib.rnn.tensor.network import RecurrentNetwork

class GaussianDTRNNPolicy(GaussianRNNPolicy):
    def create_mean_network(self, input_shape, output_dim, hidden_dim,
                            hidden_nonlinearity=NL.tanh,
                            output_nonlinearity=None, **kwargs):

        return RecurrentNetwork(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            **kwargs
        )
