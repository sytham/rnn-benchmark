import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
import theano

from rllab.core.lasagne_layers import OpLayer
from contrib.rnn.tensor.layers import RecurrentLayer, EulerCTRNNLayer, MidPointCTRNNLayer, ReparamEulerCTRNNLayer, EchoStateLayer
from rllab.misc.overrides import overrides
       

class RecurrentNetwork(object):
    def __init__(self, input_shape, output_dim, hidden_dim, hidden_nonlinearity=NL.tanh,
                 output_nonlinearity=None, name=None, input_var=None, input_layer=None, **kwargs):
        if input_layer is None:
            l_in = L.InputLayer(shape=(None, None) + input_shape, input_var=input_var, name="input")
        else:
            l_in = input_layer
        l_step_input = L.InputLayer(shape=(None,) + input_shape)
        l_step_prev_hidden = L.InputLayer(shape=(None, hidden_dim))
        l_recurrent = self.create_recurrent_layer(l_in, hidden_dim, hidden_nonlinearity=hidden_nonlinearity, **kwargs)
        l_recurrent_flat = L.ReshapeLayer(
            l_recurrent, shape=(-1, hidden_dim)
        )
        l_output_flat = L.DenseLayer(
            l_recurrent_flat,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
        )
        l_output = OpLayer(
            l_output_flat,
            op=lambda flat_output, l_input: flat_output.reshape((l_input.shape[0], l_input.shape[1], -1)),
            shape_op=lambda flat_output_shape, l_input_shape: (l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
            extras=[l_in]
        )
        l_step_hidden = l_recurrent.get_step_layer(l_step_input, l_step_prev_hidden)
        l_step_output = L.DenseLayer(
            l_step_hidden,
            num_units=output_dim,
            nonlinearity=output_nonlinearity,
            W=l_output_flat.W,
            b=l_output_flat.b,
        )
        #print(theano.printing.debugprint(L.get_output(l_step_output)))
        self._l_in = l_in
        self._hid_init_param = l_recurrent.h0
        self._l_recurrent = l_recurrent
        self._l_out = l_output
        self._l_step_input = l_step_input
        self._l_step_prev_hidden = l_step_prev_hidden
        self._l_step_hidden = l_step_hidden
        self._l_step_output = l_step_output
    
    def create_recurrent_layer(self, l_in, hidden_dim, **kwargs):
        return RecurrentLayer(l_in, num_units=hidden_dim,
                              hidden_nonlinearity=kwargs.pop('hidden_nonlinearity', NL.tanh),
                              hidden_init_trainable=True, **kwargs)
    
    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_hidden_layer(self):
        return self._l_step_prev_hidden

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

# this is a misnomer, but preserve it for pickling compatibility reasons  
class EulerCTRNN(RecurrentNetwork):
    def __init__(self, input_shape, output_dim, hidden_dim, dt,
                 hidden_nonlinearity=NL.tanh, output_nonlinearity=None, name=None, input_var=None, input_layer=None,
                 timeconstant=LI.Constant(1.), timeconstant_trainable=False,
                 **kwargs):
        self.dt = dt
        super(EulerCTRNN, self).__init__(input_shape, output_dim, hidden_dim,
                                         hidden_nonlinearity=hidden_nonlinearity, output_nonlinearity=output_nonlinearity,
                                         name=name, input_var=input_var, input_layer=input_layer,
                                        timeconstant=timeconstant, timeconstant_trainable=timeconstant_trainable,
                                        **kwargs)
    
    @overrides
    def create_recurrent_layer(self, l_in, hidden_dim, reparameterize=False, integrator='euler', **kwargs):
        if reparameterize:
            return ReparamEulerCTRNNLayer(l_in, num_units=hidden_dim,
                              hidden_nonlinearity=kwargs.pop('hidden_nonlinearity', NL.tanh),
                              dt=self.dt, hidden_init_trainable=True, **kwargs)
        if integrator.lower() == 'midpoint':
            print("Creating midpoint integrator")
            return MidPointCTRNNLayer(l_in, num_units=hidden_dim,
                              hidden_nonlinearity=kwargs.pop('hidden_nonlinearity', NL.tanh),
                              dt=self.dt, hidden_init_trainable=True, **kwargs)
        return EulerCTRNNLayer(l_in, num_units=hidden_dim,
                              hidden_nonlinearity=kwargs.pop('hidden_nonlinearity', NL.tanh),
                              dt=self.dt, hidden_init_trainable=True, **kwargs)

class EchoStateNetwork(RecurrentNetwork):    
    @overrides
    def create_recurrent_layer(self, l_in, hidden_dim, **kwargs):
        return EchoStateLayer(l_in, num_units=hidden_dim,
                              hidden_nonlinearity=kwargs.pop('hidden_nonlinearity', NL.tanh),
                              hidden_init_trainable=True, **kwargs)



    