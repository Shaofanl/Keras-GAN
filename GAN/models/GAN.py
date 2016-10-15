import numpy as np
np.set_printoptions(precision=3, suppress=True) 

from Generator import Generator
from Discriminator import Discriminator

from keras.models import Sequential, Model

from ..utils.data import floatX, transform, inverse_transform
from ..utils.vis import vis_grid
from ..utils.loss import fake_generate_loss, cross_entropy_loss
from keras.optimizers import Adam, SGD, RMSprop

from tqdm import tqdm 

class GAN(Sequential):
    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.add(generator)
        self.add(discriminator)
        
        self.generator = generator
        self.discriminator = discriminator

    def fit(self, data_stream, 
                nvis=200, 
                niter=1000,
                nbatch=128,
                nmax=None,
                k=2,
                opt=None,
                save_dir='./'):
        if nmax is None: nmax = nbatch*100
        if opt == None: opt = Adam(lr=0.0001)
        gen, dis, gendis = self.generator, self.discriminator, self
        dis.trainable = False # must prevent dis from updating
        gendis.compile(optimizer=opt, loss='binary_crossentropy')#fake_generate_loss) # same effect when y===1
        dis.trainable = True
        dis.compile(optimizer=opt, loss='binary_crossentropy')#cross_entropy_loss)

        vis_grid(data_stream().next(), (10, 10), '{}/sample.png'.format(save_dir))
        sample_zmb = floatX(np.random.uniform(-1., 1., size=(nvis, gen.g_nb_noise)))

        for iteration in range(niter):
            samples = gen.generate(sample_zmb)
            vis_grid(inverse_transform(samples), (10, 10), '{}/{}.png'.format(save_dir, iteration))

            ccc = 0
            n_updates = 0
            for real_img in tqdm(data_stream(), total=nmax/nbatch):
                ccc += 1
                if ccc > nmax/nbatch: break

                Z = floatX(np.random.uniform(-1., 1., size=(nbatch, gen.g_nb_noise)))
                if n_updates % (k+1) == 0:
                    y = np.array([[1] * nbatch]).reshape(-1, 1)
                    
                    #print 'before', dis.predict(samples).sum() 
                    g_loss = gendis.train_on_batch(Z, y)
                    #print 'after', dis.predict(samples).sum() 
                    g_loss = float(g_loss)
#                   print 'g_loss:', g_loss, 'acc: {}/{}'.format((gendis.predict(Z).round() == y).sum(), nbatch), 'mean', gendis.predict(Z).mean()
#                   vis_grid(inverse_transform(gen.predict(Z)), (10, 10), '{}/{}.g_train.png'.format(save_dir, iteration))
                else:
                    gen_img = gen.predict(Z)
                    real_img = transform(real_img)

                    X = np.concatenate((real_img, gen_img))
                    y = np.array([[1] * nbatch + [0] *nbatch]).reshape(-1, 1)
                    # keep the batch size <-- deprecated, case error because batchnormalization simplify the judgement
                   #ind = np.random.choice(nbatch*2, size=nbatch)
                   #X = X[ind]
                   #y = y[ind]

#                   print 'before', dis.predict(gen_img).mean()
#                   print 'before', dis.predict(X[y.flatten()==0]).mean()
                    d_loss = dis.train_on_batch(X[:nbatch], y[:nbatch]) + dis.train_on_batch(X[nbatch:], y[nbatch:])
                    d_loss = float(d_loss)
#                   print 'after ', dis.predict(gen_img).mean()
#                   print 'after ', dis.predict(X[y.flatten()==0]).mean()

#                   pred = dis.predict(X[y.flatten()==0])
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == 0).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten()
#                   pred = dis.predict(X[y.flatten()==1])
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == 1).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten()
#                   pred = dis.predict(X)
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == y).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten() 
                    
                    # batch normalization:
                    #   if all inputs are from generated data, or all inputs are from real data, the performance is extremly poor
                    #   if the inputs are mixed with generated and real data, the performance is best

#                   vis_grid(inverse_transform(np.concatenate((real_img[:50], gen_img[:50]))), (10, 10), '{}/{}.d_train.png'.format(save_dir, iteration))
                n_updates += 1
            print 'n_epochs=%.0f, g_loss=%.4f, d_loss=%.4f\n'%(iteration, g_loss, d_loss)

            if iteration% 5 == 0:
                gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
                dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)



