from keras import backend as K
from keras.layers.core import Layer
import numpy as np
import theano
import theano.tensor as T

def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)

class BN(Layer):
    def __init__(self, gamma_beta=True, a=1., e=1e-8, **kwargs):
        self.gamma_beta = gamma_beta
        self.a = a
        self.e = e
        super(BN, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.trainable_weights = []
        if self.gamma_beta:
            m = input_shape[1]
            self.gamma = sharedX(np.random.normal(loc=1.0, scale=0.02, size=(m,)), name=self.name+'_gamma')
            self.beta = sharedX(np.zeros((m,)), name=self.name+'_beta')
            self.trainable_weights.extend([self.gamma, self.beta])

    def call(self, x, mask=None):
        if x.ndim == 4:
            u = T.mean(x, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            s = T.mean(T.sqr(x - u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
            if self.a != 1:
                u = (1. - self.a)*0. + self.a*u
                s = (1. - self.a)*1. + self.a*s
            x = (x - u) / T.sqrt(s + self.e)
            if self.gamma_beta:
                x = x*self.gamma.dimshuffle('x', 0, 'x', 'x') + self.beta.dimshuffle('x', 0, 'x', 'x')
        elif x.ndim == 2:
            u = T.mean(x, axis=0)
            s = T.mean(T.sqr(x - u), axis=0)
            if self.a != 1:
                u = (1. - self.a)*0. + self.a*u
                s = (1. - self.a)*1. + elf.a*s
            x = (x - u) / T.sqrt(s + self.e)
            if self.gamma_beta:
                x = x*self.gamma + self.beta
        else:
            raise NotImplemented
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


#   def batchnorm(X, g=None, b=None, u=None, s=None, a=1., e=1e-8):
#       """
#       batchnorm with support for not using scale and shift parameters
#       as well as inference values (u and s) and partial batchnorm (via a)
#       will detect and use convolutional or fully connected version
#       """
#       if X.ndim == 4:
#           if u is not None and s is not None:
#               b_u = u.dimshuffle('x', 0, 'x', 'x')
#               b_s = s.dimshuffle('x', 0, 'x', 'x')
#           else:
#               b_u = T.mean(X, axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
#               b_s = T.mean(T.sqr(X - b_u), axis=[0, 2, 3]).dimshuffle('x', 0, 'x', 'x')
#           if a != 1:
#               b_u = (1. - a)*0. + a*b_u
#               b_s = (1. - a)*1. + a*b_s
#           X = (X - b_u) / T.sqrt(b_s + e)
#           if g is not None and b is not None:
#               X = X*g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')
#       elif X.ndim == 2:
#           if u is None and s is None:
#               u = T.mean(X, axis=0)
#               s = T.mean(T.sqr(X - u), axis=0)
#           if a != 1:
#               u = (1. - a)*0. + a*u
#               s = (1. - a)*1. + a*s
#           X = (X - u) / T.sqrt(s + e)
#           if g is not None and b is not None:
#               X = X*g + b
#       else:
#           raise NotImplementedError
#       return X


