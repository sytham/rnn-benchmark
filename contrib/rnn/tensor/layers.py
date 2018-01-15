import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as NL
import theano.tensor as TT
import theano

from rllab.misc import ext
from contrib.rnn.tensor.init import SpectralRadius

class RecurrentLayer(L.Layer):
    """
    Base recurrent layer
    """

    def __init__(self, incoming, num_units, hidden_nonlinearity,
                 name=None,
                 W_init=LI.GlorotUniform(), b_init=LI.Constant(0.), Wi_init=LI.GlorotUniform(),
                 hidden_init=LI.Constant(0.), hidden_init_trainable=True, **kwargs):

        if hidden_nonlinearity is None:
            hidden_nonlinearity = NL.identity

        super(RecurrentLayer, self).__init__(incoming, name=name)

        input_shape = self.input_shape[2:]

        input_dim = ext.flatten_shape_dim(input_shape)
        # self._name = name
        # initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights from input to hidden
        self.W_xh = self.add_param(Wi_init, (input_dim, num_units), name="W_xh")
        self.b_h = self.add_param(b_init, (num_units,), name="b_h", regularizable=False)
        # Recurrent weights
        self.W_hh = self.add_param(W_init, (num_units, num_units), name="W_hh")

        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity

    def step(self, x, hprev):
        h = self.nonlinearity(x.dot(self.W_xh) + hprev.dot(self.W_hh) + self.b_h)
        return h.astype(theano.config.floatX)

    def get_step_layer(self, l_in, l_prev_hidden):
        return RecurrentStepLayer(incomings=[l_in, l_prev_hidden], recurrent_layer=self)

    def get_output_shape_for(self, input_shape):
        n_batch, n_steps = input_shape[:2]
        return n_batch, n_steps, self.num_units

    def get_output_for(self, input, **kwargs):
        print("Get output for rec layer")
        n_batches = input.shape[0]
        n_steps = input.shape[1]
        input = TT.reshape(input, (n_batches, n_steps, -1))
        h0s = TT.tile(TT.reshape(self.h0, (1, self.num_units)), (n_batches, 1))
        # flatten extra dimensions
        shuffled_input = input.dimshuffle(1, 0, 2)
        hs, _ = theano.scan(fn=self.step, sequences=[shuffled_input], outputs_info=h0s)
        shuffled_hs = hs.dimshuffle(1, 0, 2)
        return shuffled_hs


class EulerCTRNNLayer(RecurrentLayer):
    def __init__(self, incoming, num_units, hidden_nonlinearity, dt, name=None,
             W_init=LI.GlorotUniform(), b_init=LI.Constant(0.),
             hidden_init=LI.Constant(0.), hidden_init_trainable=True,
             timeconstant=LI.Constant(1.), timeconstant_trainable=False,
             **kwargs):
        
        super(EulerCTRNNLayer, self).__init__(incoming, num_units, hidden_nonlinearity, **kwargs)
        self.tc = self.add_param(timeconstant, (num_units,), name="tc",
                                 trainable=timeconstant_trainable, regularizable=False)
        self.dt = dt
                    
    def step(self, x, hprev):
        h = hprev + (-hprev + RecurrentLayer.step(self, x, hprev))*self.dt / self.tc
        return h.astype(theano.config.floatX)

class ReparamEulerCTRNNLayer(EulerCTRNNLayer):                   
    def get_tc(self):
        return TT.exp(TT.log(self.tc - self.dt)) + self.dt
    def step(self, x, hprev):
        h = hprev + (-hprev + RecurrentLayer.step(self, x, hprev))*self.dt / self.get_tc()
        return h.astype(theano.config.floatX)

class MidPointCTRNNLayer(RecurrentLayer):
    def __init__(self, incoming, num_units, hidden_nonlinearity, dt, name=None,
             W_init=LI.GlorotUniform(), b_init=LI.Constant(0.),
             hidden_init=LI.Constant(0.), hidden_init_trainable=True,
             timeconstant=LI.Constant(1.), timeconstant_trainable=False,
             **kwargs):
        
        super(MidPointCTRNNLayer, self).__init__(incoming, num_units, hidden_nonlinearity, **kwargs)
        self.tc = self.add_param(timeconstant, (num_units,), name="tc",
                                 trainable=timeconstant_trainable, regularizable=False)
        self.dt = dt
                    
    def step(self, x, hprev):
        delta_h1 = (-hprev + RecurrentLayer.step(self, x, hprev))*self.dt / self.tc
        hprobe = hprev + delta_h1
        delta_h2 = (-hprobe + RecurrentLayer.step(self, x, hprobe))*self.dt / self.tc
        h = hprev + 0.5*delta_h1 + 0.5*delta_h2
        return h.astype(theano.config.floatX)
    
class EchoStateLayer(RecurrentLayer):
    def __init__(self, incoming, num_units, hidden_nonlinearity, name=None,
             W_init=SpectralRadius(density=0.2),
             hidden_init=LI.Constant(0.), hidden_init_trainable=True,
             Wi_init=LI.Uniform(0.5), leak_rate=0.5, **kwargs):
        
        if hidden_nonlinearity is None:
            hidden_nonlinearity = NL.identity

        L.Layer.__init__(self, incoming, name=name) # skip direct parent, we'll do all that init here

        input_shape = self.input_shape[2:]

        input_dim = ext.flatten_shape_dim(input_shape)
        # self._name = name
        # initial hidden state
        self.h0 = self.add_param(hidden_init, (num_units,), name="h0", trainable=hidden_init_trainable,
                                 regularizable=False)
        # Weights from input to hidden
        self.W_xh = self.add_param(Wi_init, (input_dim, num_units), name="W_xh", trainable=False, regularizable=False)
        
        # Recurrent weights
        self.W_hh = self.add_param(W_init, (num_units, num_units), name="W_hh", trainable=False, regularizable=False)
        self.leak_rate = leak_rate
        
        self.num_units = num_units
        self.nonlinearity = hidden_nonlinearity
                    
    def step(self, x, hprev):
        h = (1 - self.leak_rate)*hprev + self.leak_rate*self.nonlinearity(x.dot(self.W_xh) + hprev.dot(self.W_hh))
        return h.astype(theano.config.floatX)
    
class RecurrentStepLayer(L.MergeLayer):
    def __init__(self, incomings, recurrent_layer, name=None):
        super(RecurrentStepLayer, self).__init__(incomings, name)
        self._recurrent_layer = recurrent_layer

    def get_params(self, **tags):
        return self._recurrent_layer.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batch = input_shapes[0]
        return n_batch, self._recurrent_layer.num_units

    def get_output_for(self, inputs, **kwargs):
        x, hprev = inputs
        n_batch = x.shape[0]
        x = x.reshape((n_batch, -1))
        return self._recurrent_layer.step(x, hprev)