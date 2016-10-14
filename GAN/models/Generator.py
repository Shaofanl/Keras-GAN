import numpy as np

from keras.layers import Dense, Activation, Reshape, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
import keras

from ..layers import Deconvolution2D

class Generator(Sequential):
    def __init__(self, 
                    g_size=(3, 128, 64),
                    g_nb_filters=128,
                    g_nb_noise=200,
                **kwargs):
        super(Generator, self).__init__(**kwargs)
        
        self.g_size = g_size
        self.g_nb_filters = g_nb_filters
        self.g_nb_noise = g_nb_noise 

        c, h, w = g_size # h and w should be multiply of 16
        nf = g_nb_filters
        self.add( Dense(nf*8 * (h/16) * (w/16), input_shape=(g_nb_noise,)) )
        self.add( BatchNormalization(mode=2) )
        self.add( Activation('relu') )
        self.add( Reshape((nf*8, h/16, w/16)) )

        self.add( Deconvolution2D(nf*4, 5, 5, subsample=(2,2), border_mode=(2,2)) )
        self.add( BatchNormalization(mode=2) )
        self.add( Activation('relu') )

        self.add( Deconvolution2D(nf*2, 5, 5, subsample=(2,2), border_mode=(2,2)) )
        self.add( BatchNormalization(mode=2) )
        self.add( Activation('relu') )
        
        self.add( Deconvolution2D(nf, 5, 5, subsample=(2,2), border_mode=(2,2)) )
        self.add( BatchNormalization(mode=2) )
        self.add( Activation('relu') )

        self.add( Deconvolution2D(c, 5, 5, subsample=(2,2), border_mode=(2,2)) )
        self.add( Activation('tanh') )

    def generate(self, x):
        return self.predict(x) 

    def random_generate(self, batch_size=128):
        return self.predict(np.random.random((batch_size, self.g_nb_noise)) )

