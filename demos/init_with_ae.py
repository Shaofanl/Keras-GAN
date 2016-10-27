import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
#os.environ['THEANO_FLAGS']='lib.cnmem=0,device=gpu0'

import keras
keras.backend.theano_backend._set_device('dev0') 

import numpy as np
from sklearn.datasets import fetch_mldata

from GAN.models import Generator, Discriminator, Autoencoder, GAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
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
# init with ae and then run gan
    nbatch = 128
    x, y, stream = get_mnist(nbatch)


    g = Generator(g_size=(1, 28, 28), g_nb_filters=64, g_nb_coding=200, g_scales=2, g_FC=[1024], g_init=InitNormal(scale=0.05))
    d = Discriminator(d_size=g.g_size, d_nb_filters=64, d_scales=2, d_FC=[1024], d_init=InitNormal(scale=0.05))

    g.load_weights('models/mnist_ae_g.h5')
    print g.get_weights()[0].sum()
    d.load_weights('models/mnist_ae_d.h5')
    print d.get_weights()[0].sum()

    gan = GAN(g, d)
    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(stream, 
                save_dir='./samples/mnist', 
                k=1, 
                nbatch=nbatch,
                opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5))


    

