from keras import backend as K
from keras.engine.topology import Layer
import theano.tensor as T

class Broadcast(Layer):
    def __init__(self, dimshuffle, **kwargs):
        self.dimshuffle = dimshuffle
        super(Broadcast, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return x.dimshuffle(*self.dimshuffle) 

    def get_output_shape_for(self, input_shape):
        shape = []
        for i in self.dimshuffle:
            if i == 'x':
                shape.append(None)
            elif isinstance(i, int):
                shape.append(input_shape[i])
            else:
                raise Exception
        return tuple(shape)

