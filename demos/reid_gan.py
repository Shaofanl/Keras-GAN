import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=1,device=gpu0'

import sys
sys.path.insert(0, '/home/shaofan/.local/lib/python2.7/site-packages/')

import keras
keras.backend.theano_backend._set_device('dev0')

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import Generator, Discriminator, GAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal

if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 64, 128

    from load import people
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)


    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=128, g_nb_coding=500, g_scales=4, g_init=InitNormal(scale=0.002))#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=128, d_scales=4, d_init=InitNormal(scale=0.002))#, d_FC=[5000])
    gan = GAN(g, d)

    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(tr_stream, 
                save_dir='./samples/cuhk01/', 
                k=1, 
                nbatch=nbatch,
                nmax=nmax,
                opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5))
                #opt=RMSprop(lr=0.01))
    

