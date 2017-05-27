import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, help="cuda", default='2')
args = parser.parse_args()
print args

import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',gpuarray.preallocate=0.00,device=cuda{}'.format(args.cuda)
os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(args.cuda)


import keras
import numpy as np

from GAN.models import fc_gen, fc_dis, iWGAN
from GAN.utils.init_utils import InitNormal
from keras.datasets import mnist


class mnist_stream():
    def __init__(self):
        (x_train, _), (_, _) = mnist.load_data()
        x = np.expand_dims(x_train/255., 1)
        self.x = x

    def __call__(self, bs):
        return self.x[np.random.choice(self.x.shape[0], replace=False, size=(bs,))]


if __name__ == '__main__':
    coding = 200
    img_shape = (1, 28, 28)

    g = fc_gen((coding,), img_shape, filters=[128, 256, 512])  
    d = fc_dis(img_shape, filters=[128, 256, 512][::-1])
    gan = iWGAN(g, d,) #init=InitNormal(scale=0.02))

    from keras.optimizers import Adam, SGD, RMSprop
    gan.fit(mnist_stream(), 
                niter=20000,
                save_dir='./quickshots/iwgan', 
                k=3,
                save_iter=100,
                nbatch=100,)


