import numpy as np

from keras.layers import Dense, Activation, Reshape, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
import keras

from ..utils.init import InitNormal
from ..layers import  BN 

class Discriminator(Sequential):
    def __init__(self, 
                    d_size=(3, 128, 64),
                    d_nb_filters=128,
                    d_scales=4,
                    d_FC=None,
                **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        
        self.d_size = d_size
        self.d_nb_filters = d_nb_filters
        self.d_scales = d_scales
        self.d_FC = d_FC

        c, h, w = d_size # h and w should be multiply of 16
        nf = d_nb_filters

        for s in range(d_scales):
            if s == 0: 
                self.add( ZeroPadding2D((2, 2), input_shape=d_size) )
            else:
                self.add( ZeroPadding2D((2, 2)) )

            self.add( Convolution2D(nf*(2**s), 5, 5, subsample=(2,2), border_mode='valid', init=InitNormal) )
            self.add( BN() )
#           self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2, axis=1) )
            self.add( LeakyReLU(0.2) )

        self.add( Flatten() )
        if d_FC is not None:
            for fc_dim in d_FC:
                self.add( Dense(fc_dim, init=InitNormal, activation='relu') )
                self.add( BN() )
#               self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2) )
                self.add( Activation('relu') )
        self.add( Dense(1, activation='sigmoid', init=InitNormal) )

    def discriminate(self, x):
        return self.predict(x) 

