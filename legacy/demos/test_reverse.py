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

    vis_grid(inverse_transform(g.random_generate(100)), (10, 10), 'sample.png')
    g.load_weights('{}/{}_gen_params.h5'.format('samples/mnist_bck/', 600))
    vis_grid(inverse_transform(g.random_generate(100)), (10, 10), 'sample2.png')

    img = g.random_generate(128)
    reverse(g, img, savedir='reverse/mnist/')

   
    def reverse(self, target, min=-1.0, max=+1.0, savedir=None): 
        import theano
        import theano.tensor as T
        from ..utils.vis import vis_grid
        from ..utils.data import transform, inverse_transform

        if not hasattr(self, '_reverse_grad'):
            print 'building reverse model ...'
            Y = T.tensor4()
            I = K.placeholder(ndim=2)
            O = self(I)

            loss = T.sum((Y-O)**2)
            self._reverse_grad = theano.function([I, Y], [loss, T.grad(loss, I)])

        lr = 1e-5
        Zval = np.random.uniform(min, max, (target.shape[0], self.g_nb_coding)).astype('float32')
        lastZ = Zval.copy(); last_loss = np.inf
        for i in range(500):
            Zval = np.clip(Zval, min, max)

            if savedir:
                vis_grid(inverse_transform(np.concatenate([ self.predict(Zval[:10]), target[:10] ], axis=0)), (2, 10), 
                        '{}/{}.png'.format(savedir, i))

            # Xiao jianxiong <---
            #
            loss, grad = self._reverse_grad(Zval, target)
            print loss
            if loss < last_loss: 
                lastZ = Zval.copy()
                Zval = Zval - grad * lr 
                last_loss = loss
                lr *= 1.05
                equ_count = 0
            else:
#               if loss == last_loss: equ_count += 1
#               if equ_count > 20: break

                Zval = lastZ.copy()
                lr = lr * 0.5
        return Zval

