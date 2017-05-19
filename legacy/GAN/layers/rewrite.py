from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, output_dim, init, **kwargs):
        self.output_dim = output_dim
        self.init = init
        super(Dense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]

        self.W = self.init((input_dim, self.output_dim))
        self.b = self.init((self.output_dim))
        self.trainable_weights = [self.W, self.b]

    def call(self, x, mask=None):
        return K.dot(x, self.W) + self.b.dimshuffle('x', 0)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

