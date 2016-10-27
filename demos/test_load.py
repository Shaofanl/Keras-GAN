import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=0,device=gpu0'

import keras
keras.backend.theano_backend._set_device('dev0') 

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import Generator, Discriminator, GAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal

if __name__ == '__main__':
    g = Generator(g_size=(1, 28, 28), g_nb_filters=64, g_nb_coding=200, g_scales=2, g_FC=[1024], g_init=InitNormal(scale=0.05))
    d = Discriminator(d_size=g.g_size, d_nb_filters=64, d_scales=2, d_FC=[1024], d_init=InitNormal(scale=0.05))
    gan = GAN(g, d)

    import ipdb
    ipdb.set_trace()

    vis_grid(inverse_transform(g.random_generate(100)), (10, 10), 'sample.png')
    g.load_weights('{}/{}_gen_params.h5'.format('samples/mnist_bck/', 600))
    vis_grid(inverse_transform(g.random_generate(100)), (10, 10), 'sample2.png')

   

