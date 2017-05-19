import os
#os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,contexts=dev0->cuda0'
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',lib.cnmem=0,device=gpu0'

# import sys
# sys.path.insert(0, '~/.local/lib/python2.7/site-packages/')

import keras
print keras
keras.backend.theano_backend._set_device(None) 
print keras.backend.theano_backend.theano.config.cuda.root

import numpy as np

import sys
sys.path.insert(0, './')

from GAN.models import Generator, Discriminator, Autoencoder, GAN, AEGAN
from GAN.utils import vis_grid
from GAN.utils.data import transform, inverse_transform
from GAN.utils.init import InitNormal

from keras.optimizers import Adam, SGD, RMSprop

def feature(aegan, filename):
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        aegan.load(prefix='./samples/reid_aegan/aegan/50')

        paths = map(lambda x: x.strip(), open('protocol/cuhk01-all.txt').readlines())
        x = transform( np.array([load_image(path, (64, 128)) for path in paths]) )
        code = aegan.autoencoder.encoder.predict(x)

    ipdb.set_trace()

def test(aegan, prefix):
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        aegan.load(prefix=prefix)

        from GAN.utils.vis import vis_grid
        vis_grid(inverse_transform(aegan.generator.random_generate(128)), (2, 20), 'random_generate.png')

        paths = map(lambda x: x.strip(), open('protocol/cuhk01-all.txt').readlines())
        from load import load_image
        sample = transform( np.array([load_image(path, (64, 128)) for path in paths[:128]]) )
        
        vis_grid(inverse_transform(sample), (2, 20), 'sample.png')
        vis_grid(inverse_transform(aegan.autoencoder.autoencoder.predict(sample)), (2, 20), 'reconstruct.png')


        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
#       codes = aegan.autoencoder.encoder.predict(sample)
#       codes = aegan.generator.sample(128)
        codes = aegan.autoencoder.encoder.predict(aegan.generator.random_generate(128))


        for ind, code in enumerate(codes):
            n, bins, patches = plt.hist(code, 50, normed=1, facecolor='green', alpha=0.75)
            plt.savefig('test/{}.pdf'.format(ind))
            plt.clf()

    ipdb.set_trace()


if __name__ == '__main__':
    nbatch = 128 
    nmax   = nbatch * 100
    npxw, npxh = 64, 128

    from load import people, load_all
    va_data, tr_stream, _ = people(pathfile='protocol/cuhk01-train.txt', size=(npxw, npxh), batch_size=nbatch)
    allx = transform(load_all('protocol/cuhk01-train.txt', (npxw, npxh)))

    g = Generator(g_size=(3, npxh, npxw), g_nb_filters=128, g_nb_coding=5000, g_scales=4, g_init=InitNormal(scale=0.002))#, g_FC=[5000])
    d = Discriminator(d_size=g.g_size, d_nb_filters=128, d_scales=4, d_init=InitNormal(scale=0.002))#, d_FC=[5000])

# init with autoencoder
    ae = Autoencoder(g, d)
#   ae.fit(tr_stream, 
#           save_dir='./samples/reid_aegan_5000/ae/',
#           nbatch=nbatch,
#           opt=Adam(lr=0.002),
#           niter=1001)
    ae.autoencoder.load_weights('./samples/reid_aegan_5000/ae/1000_ae_params.h5')

# run aegan 
    aegan = AEGAN(g, d, ae)
    aegan.load('./samples/reid_aegan_5000/aegan/60')
    import ipdb
    with ipdb.launch_ipdb_on_exception():
        aegan.fit(tr_stream, 
                    save_dir='./samples/reid_aegan_5000/aegan/', 
                    k=1, 
                    nbatch=nbatch,
                    nmax=nbatch*100,
                    opt=Adam(lr=0.0002, beta_1=0.5, decay=1e-5),
                    niter=501,)
#                   rec_with_only_dis=True)
#                   pull=False, pullx=None, pully=None)
#                   pull=True, pullx=allx, pully=None, pullcoef=5e-6)


#   test(aegan, './samples/reid_aegan_fixed/aegan/150')
#   feature(aegan, 'protocol/cuhk01-all.txt')
