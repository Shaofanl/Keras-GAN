from keras import backend as K
from keras.layers.core import Layer

import numpy as np

import theano
import theano.tensor as T
from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty
from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConv, GpuDnnConvGradI, dnn_conv, dnn_pool

class Deconvolution2D(Layer):
    '''
        A simple implement of deconvolution. A forward convolution description is created, and the gradient is used as output. Try to deduct the changes of dimension in 1-dim vector if you are confused..
    '''
    def __init__(self, nb_filter, nb_row, nb_col, subsample=(2,2), border_mode=(2,2), conv_mode='conv', init=None, **kwargs):
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.subsample = subsample
        self.border_mode = border_mode
        self.conv_mode = conv_mode
        self.init = init
        super(Deconvolution2D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, c, h, w = input_shape 
        
        if self.init == None:
            self.W = K.variable(np.random.random((c, self.nb_filter, self.nb_row, self.nb_col)), dtype=theano.config.floatX, name='{}_W'.format(self.name))
        else:
            self.W = self.init(shape=(c, self.nb_filter, self.nb_row, self.nb_col), name='{}_W'.format(self.name))
        self.trainable_weights = [self.W]
        
    def call(self, x, mask=None):
        x = gpu_contiguous(x)
        k = gpu_contiguous(self.W)
        new_size = (x.shape[0], k.shape[1], x.shape[2]*self.subsample[0], x.shape[3]*self.subsample[1])

        out = gpu_alloc_empty(*new_size)
        desc = GpuDnnConvDesc(border_mode=self.border_mode,
                              subsample=self.subsample,
                              conv_mode=self.conv_mode)(out.shape, k.shape)
        return GpuDnnConvGradI()(k, x, out, desc)

    def get_output_shape_for(self, input_shape):
        n, c, h, w = input_shape 
        return (n, self.nb_filter, h*self.subsample[0], w*self.subsample[1])



