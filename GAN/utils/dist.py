import numpy as np
import keras.backend as K
from keras.models import Sequential
from keras.layers import Lambda

# definition of distribution, used in infoGAN

class Distribution(object):
    def __init__(self):
        raise NotImplemented
        self.dim = None # required dim in coding
        self.Qdim = None # required dim in Q(c|x) 

        self.info_tensor = None # tensor for input, of dim self.dim
        self.param_tensor = None # tensor for output param, of dim self.Qdim 
        self.log_Q_c_given_x = None # a value

    def register(self, info_tensor, param_tensor):
        '''
            info_tensor:       (tensor) the coding used to generate image
            param_tensor:      (tensor) dim == self.Qdim
            log_Q_c_given_x:    (tensor) a real value
        '''
        raise NotImplemented

    def sample(self, batch_size=128, **kwargs):
        '''
            Sample from distribution
            return: (batch_size, self.dim)
        '''
        raise NotImplemented


class ProductDist(Distribution):
    def __init__(self, dists):
        self.dists = dists
        self.dim = reduce(lambda x, y: x+y.dim, dists, 0)
        self.Qdim = reduce(lambda x, y: x+y.Qdim, dists, 0)
        
    def register(self, info_tensor, param_tensor):
        self.info_tensor = info_tensor
        self.param_tensor = param_tensor

        info_shift = 0
        param_shift = 0
        self.log_Q_c_given_x = []
        for dist in self.dists:
            if not hasattr(dist, 'log_Q_c_given_x'):
                dist.register(info_tensor[:, info_shift:info_shift+dist.dim],
                              param_tensor[:, param_shift:param_shift+dist.Qdim])
            self.log_Q_c_given_x.append( dist.log_Q_c_given_x )
        return K.concatenate(self.log_Q_c_given_x, axis=1)

    def sample(self, batch_size=128, **kwargs):
        c = [dist.sample(batch_size) for dist in self.dists]
        c = np.concatenate(c, axis=1)
        return c

class CategoryDist(Distribution):
    def __init__(self, n):
        self.n = n
        self.dim = n
        self.Qdim = n
        
    def register(self, info_tensor, param_tensor):
        self.info_tensor = info_tensor
        self.param_tensor = param_tensor 
        self.log_Q_c_given_x = \
            K.sum(K.log(K.softmax(param_tensor)+K.epsilon()) * info_tensor, axis=1)
#       m = Sequential([ Activation('softmax', input_shape=(self.n,)), Lambda(lambda x: K.log(x), lambda x: x) ])
        return K.reshape(self.log_Q_c_given_x, (-1, 1))

    def sample(self, batch_size=128, **kwargs):
        _c = np.random.randint(self.n, size=(batch_size,))
        c = np.zeros((batch_size, self.n))
        c[np.arange(batch_size), _c] = 1.0
        return c


