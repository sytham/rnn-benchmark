import numpy as np

import theano.tensor as TT
import theano
import lasagne.nonlinearities as NL
from contrib.rnn.tensor.policies.base import GaussianRNNPolicy
from contrib.rnn.tensor.network import EulerCTRNN
from rllab.misc.overrides import overrides
from rllab.misc.tensor_utils import unflatten_tensors

class GaussianCTRNNPolicy(GaussianRNNPolicy):
    def create_mean_network(self, input_shape, output_dim, hidden_dim,
                            hidden_nonlinearity=NL.tanh,
                            output_nonlinearity=None,
                            dt=0.01, reparameterize=False, integrator='euler', **kwargs):

        self.dt = dt
        self.reparam = reparameterize
        return EulerCTRNN(
            input_shape=input_shape,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dt=dt,
            reparameterize=reparameterize,
            integrator=integrator,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            **kwargs
        )
    
    def get_leq_constraints(self):
        tcparam = None
        for param in self.get_params(trainable=True):
            if param.name == "tc":
                tcparam = param
                break
        if tcparam is None:
            return [] # apparently no trainable timeconstant
        
        #dt = np.tile(self.dt, self.n_hidden)
        coeff = 1.0
        constr = coeff * TT.sum(NL.rectify(self.dt - tcparam))
#         costheta = tcparam.dot(dt) / (tcparam.norm(2) * np.linalg.norm(dt))
#         constr = TT.sqrt(2)*self.dt / (costheta - TT.sqrt(1 - costheta**2 + 1e-6)) - tcparam.norm(2)
        # <= constraint, so negate
        #return [(-TT.min(tcparam), -self.dt)]
        #return [(TT.maximum(self.dt - TT.min(tcparam), 0), 0.0)]
        return [(constr.astype(theano.config.floatX), 0.0)]
    
    @overrides
    def set_param_values(self, flattened_params, **tags):
        debug = tags.pop("debug", False)
        param_values = unflatten_tensors(
            flattened_params, self.get_param_shapes(**tags))
        for param, dtype, value in zip(
                self.get_params(**tags),
                self.get_param_dtypes(**tags),
                param_values):
            if not self.reparam and param.name == "tc":
                if np.any(value < self.dt):
                    print("Tc constraint violated:", self.dt, value)
                value = np.maximum(self.dt, value)
            param.set_value(value.astype(dtype))
            if debug:
                print("setting value of %s" % param.name)

