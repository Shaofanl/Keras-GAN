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
                    d_init=None,
                **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        
        self.d_size = d_size
        self.d_nb_filters = d_nb_filters
        self.d_scales = d_scales
        self.d_FC = d_FC
        self.d_init = d_init if d_init is not None else InitNormal()

        c, h, w = d_size # h and w should be multiply of 16
        nf = d_nb_filters

        for s in range(d_scales):
            if s == 0: 
                self.add( ZeroPadding2D((2, 2), input_shape=d_size) )
            else:
                self.add( ZeroPadding2D((2, 2)) )

            self.add( Convolution2D(nf*(2**s), 5, 5, subsample=(2,2), border_mode='valid', init=self.d_init) )
            self.add( BN() )
#           self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2, axis=1) )
            self.add( LeakyReLU(0.2) )

        self.add( Flatten() )
        if d_FC is not None:
            for fc_dim in d_FC:
                self.add( Dense(fc_dim, init=self.d_init) )
                self.add( LeakyReLU(0.2) )
                self.add( BN() )
#               self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2) )
                self.add( LeakyReLU(0.2) )
        self.add( Dense(1, activation='sigmoid', init=self.d_init) )

    def discriminate(self, x):
        return self.predict(x) 


class Critic(Sequential): # as in Wasserstein GAN
    def __init__(self, 
                    d_size=(3, 128, 64),
                    d_nb_filters=128,
                    d_scales=4,
                    d_FC=None,
                    d_init=None,
                **kwargs):
        super(Critic, self).__init__(**kwargs)
        
        self.d_size = d_size
        self.d_nb_filters = d_nb_filters
        self.d_scales = d_scales
        self.d_FC = d_FC
        self.d_init = d_init if d_init is not None else InitNormal()

        c, h, w = d_size # h and w should be multiply of 16
        nf = d_nb_filters

        for s in range(d_scales):
            if s == 0: 
                self.add( ZeroPadding2D((2, 2), input_shape=d_size) )
            else:
                self.add( ZeroPadding2D((2, 2)) )

            self.add( Convolution2D(nf*(2**s), 5, 5, subsample=(2,2), border_mode='valid', init=self.d_init) )
            self.add( BN() )
#           self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2, axis=1) )
            self.add( LeakyReLU(0.2) )

        self.add( Flatten() )
        if d_FC is not None:
            for fc_dim in d_FC:
                self.add( Dense(fc_dim, init=self.d_init) )
                self.add( LeakyReLU(0.2) )
                self.add( BN() )
#               self.add( BatchNormalization(beta_init='zero', gamma_init='one', mode=2) )
                self.add( LeakyReLU(0.2) )
        self.add( Dense(1, activation='linear', init=self.d_init) )


