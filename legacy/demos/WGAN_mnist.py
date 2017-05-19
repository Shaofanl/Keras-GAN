import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,device=gpu0'
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

from models_WGAN import generator_upsampling, generator_deconv
from models_WGAN import discriminator

if __name__ == '__main__':
    nbatch = 64 
    x, y, stream = get_mnist(nbatch)

    init = InitNormal(scale=0.02)
#   g = MLP(g_size=(1, 28, 28), 
#               g_nb_filters=128, 
#               g_nb_coding=50, 
#               g_init=init)
    g = Generator(g_size=(1, 28, 28), g_nb_filters=32, g_FC=[1024],
                g_nb_coding=100, g_scales=2, g_init=init)


#   d = Sequential([
#           Flatten(input_shape=g.output_shape[1:]),
#           Dense(128, init=init),
#           Activation('relu'),
#           Dense(128, init=init),
#           Activation('relu'),
#           Dense(128, init=init),
#           Activation('relu'),
#           Dense(1, init=init),
#       ])
    d = Critic(d_size=(1, 28, 28), d_nb_filters=32, d_FC=[1024], d_scales=2, d_init=init)
    gan = WGAN(g, d)


#   g = generator_upsampling(noise_dim=(100,),
#               img_dim=(1, 28, 28), 
#               bn_mode=2, 
#               model_name="generator_upsampling", 
#               dset="mnist")
#   g.g_nb_coding = 100 
#   def generate(x):
#       return g.predict(x) 
#   def random_generate(batch_size=128):
#       return g.predict(g.sample(batch_size))
#   def sample(batch_size=128):
#       return np.random.normal(0, 0.5, size=(batch_size, g.g_nb_coding))
#   g.generate = generate
#   g.random_generate = random_generate
#   g.sample = sample
#   d = discriminator(img_dim=(1,28,28), bn_mode=2,model_name='discriminator')
#   gan = DCGAN(g, d, (64,), (1, 28, 28))


    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(stream, save_dir='./labs/WGAN_mnist',
        opts = [RMSprop(lr=5e-5),
                RMSprop(lr=5e-5)],
        niter=100000,
        plot_iter=200,
    )

# 10/26/16: if not initialize in a proper way
#           all-zero will appears in BN layer and cause nan
#           solution: use leaky relu


    
