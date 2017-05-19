import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=1,device=gpu0'

import sys
sys.path.insert(0, '/home/shaofan/.local/lib/python2.7/site-packages/')

import keras
keras.backend.theano_backend._set_device('dev0')

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.utils.init import InitNormal
from GAN.models import InfoGenerator, Discriminator, InfoGAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.dist import CategoryDist , UniformDist

from keras.models import Sequential
from keras.layers import Activation, Input, Dense, Activation
from GAN.layers import BN



if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 32, 64 #64, 128

    from load import people
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)

    g = InfoGenerator(g_size=(3, npxh, npxw), 
                      g_nb_filters=128, 
                      g_nb_noise=500,  # not coding but noise
                      g_scales=3, 
                      g_FC=None,
                      g_info=[
#                               CategoryDist(n=10, lmbd=1e-3), 
                                UniformDist(min=-1, max=+1, lmbd=1e-3),
                             ],
                      g_init=InitNormal(scale=0.002)
                      )
    d = Discriminator(d_size=g.g_size, 
                        d_nb_filters=128, 
                        d_scales=3, 
                        d_FC=None, 
                        d_init=InitNormal(scale=0.002))
    Q = Sequential([ Dense(500, batch_input_shape=d.layers[-2].output_shape) ,
                     BN(),
                     Activation('relu'),
                     Dense(g.g_info.Qdim),
                   ])
    gan = InfoGAN(generator=g, discriminator=d, Qdist=Q)


    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(tr_stream, 
                save_dir='./samples/cuhk01_info_small/', 
                k=1, 
                nbatch=nbatch,
                nmax=nmax,
                opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5))
                #opt=RMSprop(lr=0.01))
    

