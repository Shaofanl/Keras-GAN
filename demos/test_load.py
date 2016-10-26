import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=1,device=gpu0'

import sys
sys.path.insert(0, '/home/shaofan/.local/lib/python2.7/site-packages/')

import numpy as np
import keras
keras.backend.theano_backend._set_device('dev0')

from GAN.models import Generator, Discriminator, GAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal

if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 32, 64 

    from load import people
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)

    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=64, g_nb_coding=500, g_scales=3, g_init=InitNormal(scale=0.002))#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=64, d_scales=3, d_init=InitNormal(scale=0.002))#, d_FC=[5000])
    gan = GAN(g, d)

    import ipdb
    ipdb.set_trace()

    g.load_weights('samples/cuhk01_bck/50_gen_params.h5')

