import numpy as np

from keras.layers import Dense, Activation, Reshape, Flatten, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential, Model
import keras

class Discriminator(Sequential):
    def __init__(self, 
                    d_size=(3, 128, 64),
                    d_nb_filters=128,
                **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        
        self.d_size = d_size
        self.d_nb_filters = d_nb_filters

        c, h, w = d_size # h and w should be multiply of 16
        nf = d_nb_filters

        self.add( ZeroPadding2D((2, 2), input_shape=d_size) )
        self.add( Convolution2D(nf, 5, 5, subsample=(2,2), border_mode='valid') )
        self.add( BatchNormalization(mode=2) )
        self.add( LeakyReLU() )

        self.add( ZeroPadding2D((2, 2)) )
        self.add( Convolution2D(nf*2, 5, 5, subsample=(2,2), border_mode='valid') )
        self.add( BatchNormalization(mode=2) )
        self.add( LeakyReLU() )

        self.add( ZeroPadding2D((2, 2)) )
        self.add( Convolution2D(nf*4, 5, 5, subsample=(2,2), border_mode='valid') )
        self.add( BatchNormalization(mode=2) )
        self.add( LeakyReLU() )

        self.add( ZeroPadding2D((2, 2)) )
        self.add( Convolution2D(nf*8, 5, 5, subsample=(2,2), border_mode='valid') )
        self.add( BatchNormalization(mode=2) )
        self.add( LeakyReLU() )

        self.add( Flatten() )
        self.add( Dense(1, activation='sigmoid') )

    def discriminate(self, x):
        return self.predict(x) 

