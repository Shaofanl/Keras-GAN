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


    g = Generator(g_size=(1, 28, 28), g_nb_filters=64, g_nb_coding=100, g_scales=2, g_FC=[1024])
    d = Discriminator(d_size=g.g_size, d_nb_filters=64, d_scales=2, d_FC=[1024])
    gan = GAN(g, d)
    gan.fit(stream, save_dir='./samples/mnist', k=1, nbatch=nbatch)
    

