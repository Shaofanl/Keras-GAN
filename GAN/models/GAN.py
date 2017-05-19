import os 
import sys
import numpy as np
from ..utils.vis_utils import vis_grid
from keras.models import Sequential, Model
from keras.layers import Input

class GAN(object):
    def __init__(self, generator, discriminator, **kwargs):
        self.generator = generator
        self.discriminator = discriminator
        self.gan = Sequential([generator, discriminator])

        self.coding = self.generator.input_shape[1]

        if 'init' in kwargs:
            init = kwargs['init']
            init(self.generator)
            init(self.discriminator)

        generator.summary()
        discriminator.summary()
        self.gan.summary()

    def fit(self, data_generator, 
                nvis=120, 
                niter=1000,
                nbatch=128,
                k=2,
                opt=None,
                save_dir='./quickshots/',
                save_iter=5):
        if opt == None: opt = Adam(lr=0.0001)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        gen, dis, gendis = self.generator, self.discriminator, self.gan

        shape = dis.get_input_shape_at(0)[1:]
        gen_input, real_input = Input(shape), Input(shape)
        dis2batch = Model([gen_input, real_input], [dis(gen_input), dis(real_input)])


        vis_grid(data_generator(60), (5, 12), '{}/sample_real.png'.format(save_dir))
        sample_zmb = np.random.uniform(-1., 1., size=(nvis, self.coding)).astype('float32')
        vis_grid(gen.predict(sample_zmb), (5, 12), '{}/sample_generate.png'.format(save_dir))

        dis.trainable = False 
        gendis.compile(optimizer=opt, loss='binary_crossentropy')
        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss='binary_crossentropy', 
                    metrics=['binary_accuracy'])

        g_loss, d_loss = 0, 0
        for iteration in range(1, niter+1):
            print 'iteration', iteration
            real_img = data_generator(nbatch)
            Z = np.random.uniform(-1., 1., size=(nbatch, self.coding)).astype('float32')

            if (k>0 and iteration%(k+1) == 0) or (k<0 and iteration%(-k+1) != 0): 
                y = np.ones((nbatch, 1))
                g_loss = gendis.train_on_batch(Z, y)
                g_loss = float(g_loss)
                print '\tg_loss=%.4f'%(g_loss)
            else:
                gen_img = gen.predict(Z)
                gen_y   = np.zeros((nbatch, 1))
                real_y  = np.ones((nbatch, 1))
                d_loss = dis2batch.train_on_batch([gen_img, real_img], [gen_y, real_y])
                d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc = d_loss
                print '\td_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc'
                print '\t%.4f %.4f %.4f %.4f %.4f'%(d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc)

            if iteration%save_iter== 0:
                samples = gen.predict(sample_zmb)
                vis_grid(samples, (5, 12), '{}/{}.png'.format(save_dir, iteration))
#               gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
#               dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)


