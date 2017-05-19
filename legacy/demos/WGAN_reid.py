import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=1.00,device=gpu0'
import sys
sys.path.insert(0, './')
sys.path.insert(0, '/home/shaofan/.local/lib/python2.7/site-packages') 

import keras
keras.backend.theano_backend._set_device(None) 

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import Generator, Critic, WGAN, MLP
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

if __name__ == '__main__':
    nbatch = 64 
    npxw, npxh = 64, 128

    from load import people
    va_data, stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)

    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=128, g_nb_coding=50, g_scales=4, g_init=InitNormal(scale=0.001))
    d = Critic(d_size=g.g_size, d_nb_filters=128, d_scales=4, d_init=InitNormal(scale=0.001))

    gan = WGAN(g, d)

    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(stream, save_dir='./labs/WGAN_reid',
        opts = [RMSprop(lr=0.00005, clipvalue=0.01),
                RMSprop(lr=0.00005)],
        niter=100000,
    )

