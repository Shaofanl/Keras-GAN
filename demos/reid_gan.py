import os
os.environ['THEANO_FLAGS']='lib.cnmem=1,contexts=dev0->cuda0,cuda.root=/home/shaofan/installs/cudnnv5/mycuda/'
#os.environ['THEANO_FLAGS']='lib.cnmem=0,device=gpu0'

import keras
keras.backend.theano_backend._set_device('dev0')

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import Generator, Discriminator, GAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform

if __name__ == '__main__':
    nbatch = 4 
    nmax   = nbatch * 100
    npxw, npxh = 64, 128

    from load import people
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)


    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=128, g_nb_coding=200, g_scales=4)#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=128, d_scales=4)#, d_FC=[5000])
    gan = GAN(g, d)

    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(tr_stream, 
                save_dir='./samples/cuhk01/', 
                k=2, 
                nbatch=nbatch,
                nmax=nmax,
                opt=RMSprop(lr=0.0005))
    

