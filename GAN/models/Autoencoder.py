import numpy as np
np.set_printoptions(precision=3, suppress=True) 

from Generator import Generator
from Discriminator import Discriminator

import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input, Dense, Activation
from keras.callbacks import LambdaCallback

from ..utils.data import floatX, transform, inverse_transform
from ..utils.vis import vis_grid
from ..utils.loss import fake_generate_loss, cross_entropy_loss, masked_loss

from tqdm import tqdm 

class Autoencoder(object):
    def __init__(self, generator, discriminator, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)

        self.ae = Sequential([layer for layer in discriminator.layers[:-1]]+\
                            [Dense(generator.g_nb_coding, activation='tanh'), generator])
        self.generator = generator
        self.discriminator = discriminator

    def fit(self, data_stream, 
                nvis=20, 
                nbatch=128,
                niter=1000,
                opt=None,
                save_dir='./'):
        if opt == None: opt = Adam(lr=0.0001)

        ae = self.ae
        ae.compile(optimizer=opt, loss='mse')

        vis_grid(data_stream().next(), (1, 20), '{}/sample.png'.format(save_dir))

        sampleX = transform(data_stream().next()[:nvis])
        vis_grid(inverse_transform(np.concatenate([sampleX, ae.predict(sampleX)], axis=0)), (2, 20), '{}/sample_generate.png'.format(save_dir))

        def vis_grid_f(epoch, logs):
            vis_grid(inverse_transform(np.concatenate([sampleX, ae.predict(sampleX)], axis=0)), (2, 20), '{}/{}.png'.format(save_dir, epoch))

        def transform_wrapper():
            for data in data_stream():
                yield transform(data), transform(data) 

        ae.fit_generator(transform_wrapper(),
                            samples_per_epoch=nbatch, 
                            nb_epoch=niter, 
                            verbose=1, 
                            callbacks=[LambdaCallback(on_epoch_end=vis_grid_f)],
                        )


 
