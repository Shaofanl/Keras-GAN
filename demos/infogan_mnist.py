import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0.9,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=1,device=gpu0'

import sys
import keras
import theano
keras.backend.theano_backend._set_device('dev0')

from keras.models import Sequential
from keras.layers import Activation, Input, Dense, Activation
from GAN.layers import BN

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import InfoGenerator, Discriminator, InfoGAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.dist import CategoryDist , UniformDist
from GAN.utils.init import InitNormal

def get_mnist(nbatch=128):
    mnist = fetch_mldata('MNIST original', data_home='/home/shaofan/.sklearn/') 
    x, y = mnist.data, mnist.target
    x = x.reshape(-1, 1, 28, 28)

    ind = np.random.permutation(x.shape[0])
    x = x[ind]
    y = y[ind]

    def random_stream():
        while 1:
            yield x[np.random.choice(x.shape[0], replace=False, size=nbatch)].transpose(0, 2, 3, 1)
    return x, y, random_stream

if __name__ == '__main__':
    nbatch = 128 
    x, y, stream = get_mnist(nbatch)

    g = InfoGenerator(g_size=(1, 28, 28), 
                      g_nb_filters=64, 
                      g_nb_noise=100,  # not coding but noise
                      g_scales=2, 
                      g_FC=[1024],
                      g_info=[
                        CategoryDist(n=10, lmbd=1e-3),
#                       UniformDist(min=-1.0,max=+1.0,lmbd=1e-3,stddev_fix=True),
#                       UniformDist(min=-1.0,max=+1.0,lmbd=1e-3,stddev_fix=True),
#                       UniformDist(min=-1.0,max=+1.0,lmbd=1e-3,stddev_fix=True),
                      ],
                      g_init=InitNormal(scale=0.02),
                      )
    d = Discriminator(d_size=g.g_size, d_nb_filters=64, d_scales=2, d_FC=[1024], d_init=InitNormal(scale=0.02))
    Q = Sequential([ Dense(200, batch_input_shape=d.layers[-2].output_shape) ,
                     BN(),
                     Activation('relu'),
                     Dense(g.g_info.Qdim),
                   ])

    gan = InfoGAN(generator=g, discriminator=d, Qdist=Q)
    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(stream, save_dir='./samples/mnist_info', k=2, 
                                        nmax=nbatch*100,
                                        nbatch=nbatch, 
#                                       opt=RMSprop(lr=0.0005))
                                        opt=Adam(lr=0.0001))
