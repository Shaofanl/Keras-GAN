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
    def __init__(self, dists, lmbd=1.0):
        self.dists = dists
        self.lmbd = lmbd
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
            self.log_Q_c_given_x.append( dist.log_Q_c_given_x.dimshuffle(0, 'x') )
        return K.concatenate(self.log_Q_c_given_x, axis=1) * self.lmbd

    def sample(self, batch_size=128, ordered=False, **kwargs):
        if ordered:
            group_size = batch_size / len(self.dists)
            if group_size * len(self.dists) < batch_size: 
                group_size += 1

            c = []
            for ordered_dist in self.dists:
                c_group = []
                for random_dist in self.dists:
                    if random_dist == ordered_dist:
                        c_group.append(ordered_dist.sample(group_size, ordered=True, **kwargs))
                    else:
                        c_group.append(random_dist.sample(group_size, ordered=False, **kwargs))
                c_group = np.concatenate(c_group, axis=1)
                c.append(c_group)
            c = np.concatenate(c, axis=0)
            c = c[:batch_size]
        else:
            c = [dist.sample(batch_size, **kwargs) for dist in self.dists]
            c = np.concatenate(c, axis=1)
        return c

class CategoryDist(Distribution):
    def __init__(self, n, lmbd=1e-3):
        self.n = n
        self.dim = n
        self.Qdim = n
        self.lmbd = lmbd
        
    def register(self, info_tensor, param_tensor):
        self.info_tensor = info_tensor
        self.param_tensor = param_tensor 
        self.log_Q_c_given_x = \
            K.sum(K.log(K.softmax(param_tensor)+K.epsilon()) * info_tensor, axis=1) * self.lmbd
#       m = Sequential([ Activation('softmax', input_shape=(self.n,)), Lambda(lambda x: K.log(x), lambda x: x) ])
        return K.reshape(self.log_Q_c_given_x, (-1, 1))

    def sample(self, batch_size=128, ordered=False, **kwargs):
        c = np.zeros((batch_size, self.n))
        if ordered:
            for ind in range(batch_size):
                c[ind][-(ind%self.n+1)] = 1.0
        else:
            _c = np.random.randint(self.n, size=(batch_size,))
            c[np.arange(batch_size), _c] = 1.0
        return c


class UniformDist(Distribution):
    '''
        treat Q(c|x) as gaussian distribution
    '''
    def __init__(self, min, max, lmbd=1e-3, stddev_fix=False):
        self.min = min
        self.max = max
        self.dim = 1
        self.stddev_fix = stddev_fix
        if stddev_fix:
            self.Qdim = 1
        else:
            self.Qdim = 2
        self.lmbd = lmbd
        
    def register(self, info_tensor, param_tensor):
        self.info_tensor = info_tensor #(128,1)

        if self.stddev_fix:
            self.param_tensor = param_tensor

            mean = K.clip(param_tensor[:, 0].dimshuffle(0, 'x'), self.min, self.max) 
            std  = 1.0
        else:
            self.param_tensor = param_tensor # 2 

            mean = K.clip(param_tensor[:, 0].dimshuffle(0, 'x'), self.min, self.max) 
          # std  = K.maximum( param_tensor[:, 1].dimshuffle(0, 'x'), 0)
            std  = K.sigmoid( param_tensor[:, 1].dimshuffle(0, 'x') )

        e = (info_tensor-mean)/(std + K.epsilon())
        self.log_Q_c_given_x = \
            K.sum(-0.5*np.log(2*np.pi) -K.log(std+K.epsilon()) -0.5*(e**2), axis=1) * self.lmbd

#       m = Sequential([ Activation('softmax', input_shape=(self.n,)), Lambda(lambda x: K.log(x), lambda x: x) ])
        return K.reshape(self.log_Q_c_given_x, (-1, 1))

    def sample(self, batch_size=128, ordered=False, orderedN=10, **kwargs):
        if ordered:
            c = np.zeros((batch_size, 1))
            intervals = np.linspace(self.min, self.max, orderedN)
            for ind in range(batch_size):
                c[ind, 0] = intervals[ind%orderedN] 
        else:
            c = np.random.uniform(self.min, self.max, size=(batch_size, 1))
        return c


