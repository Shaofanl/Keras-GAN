import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=1,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=0,device=gpu0'

import keras
keras.backend.theano_backend._set_device('dev0') 

import numpy as np

from GAN.models import Generator, Discriminator, Autoencoder, GAN, AEGAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal

from keras.optimizers import Adam, SGD, RMSprop

if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 64, 128

    from load import people
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)

    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=128, g_nb_coding=500, g_scales=4, g_init=InitNormal(scale=0.002))#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=128, d_scales=4, d_init=InitNormal(scale=0.002))#, d_FC=[5000])

# init with autoencoder
    ae = Autoencoder(g, d)
    ae.fit(stream, 
            save_dir='./samples/reid_aegan/ae/', 
            nbatch=nbatch,
            opt=Adam(lr=0.001),
            niter=501)

# run aegan 
    aegan = AEGAN(g, d)
    aegan.fit(stream, 
                save_dir='./samples/reid_aegan/aegan/', 
                k=1, 
                nbatch=nbatch,
                opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5),
                niter=501)


        

