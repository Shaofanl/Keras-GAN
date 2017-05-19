import os
#os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,device=gpu0'

import keras
print keras
keras.backend.theano_backend._set_device(None) 
print keras.backend.theano_backend.theano.config.cuda.root

import numpy as np

import sys
sys.path.insert(0, './')

from GAN.models import Generator, Discriminator, Autoencoder, GAN, AEGAN
from GAN.utils import vis_grid
from GAN.utils.data import transform_skeleton, inverse_transform_skeleton
from GAN.utils.init import InitNormal

from keras.optimizers import Adam, SGD, RMSprop

if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 64, 128

    from load import people, load_all
    va_data, tr_stream, _ = people(pathfile='protocol/PPPS.txt', size=(npxw, npxh), batch_size=nbatch)


    g = Generator(g_size=(8, npxh, npxw), g_nb_filters=128, g_nb_coding=500, g_scales=4, g_init=InitNormal(scale=0.002))#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=128, d_scales=4, d_init=InitNormal(scale=0.002))#, d_FC=[5000])
    gan = GAN(g, d)

    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(tr_stream, 
                save_dir='./samples/parsing_skeleton/', 
                k=1, 
                nbatch=nbatch,
                nmax=nmax,
                opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5),
                transform=transform_skeleton, #opt=RMSprop(lr=0.01))
                inverse_transform=inverse_transform_skeleton)
    



