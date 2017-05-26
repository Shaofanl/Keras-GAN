import os 
import sys
import numpy as np
from ..utils.vis_utils import vis_grid
from keras.models import Sequential, Model
from keras.layers import Input
from keras.optimizers import Adam

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

    def generate(self, inputs):
        return self.generator.predict(inputs)

    def build(self, **kwargs):
        opt = kwargs['opt']

        gen, dis, gendis = self.generator, self.discriminator, self.gan

        shape = dis.get_input_shape_at(0)[1:]
        gen_input, real_input = Input(shape), Input(shape)
        dis2batch = Model([gen_input, real_input], [dis(gen_input), dis(real_input)])

        dis.trainable = False 
        gendis.compile(optimizer=opt, loss='binary_crossentropy')
        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_accuracy'])

        self.gen_trainner = gendis
        self.dis_trainner = dis2batch

    def fit(self, data_generator, 
                niter=1000,
                nbatch=128,
                k=2,
                opt=None,
                save_dir='./quickshots/',
                save_iter=5):
        if opt == None: opt = Adam(lr=0.0001)
        self.build(opt=opt)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        vis_grid(data_generator(60), (5, 12), '{}/sample_real.png'.format(save_dir))
        sample_zmb = np.random.uniform(-1., 1., size=(60, self.coding)).astype('float32')
        vis_grid(self.generate(sample_zmb), (5, 12), '{}/sample_generate.png'.format(save_dir))

        g_loss, d_loss = 0, 0
        for iteration in range(1, niter+1):
            print 'iteration', iteration
            real_img = data_generator(nbatch)
            Z = np.random.uniform(-1., 1., size=(nbatch, self.coding)).astype('float32')

            if (k>0 and iteration%(k+1) == 0) or (k<0 and iteration%(-k+1) != 0): 
                y = np.ones((nbatch, 1))
                g_loss = self.gen_trainner.train_on_batch(Z, y)
                g_loss = float(g_loss)
                print '\tg_loss=%.4f'%(g_loss)
            else:
                gen_img = self.generate(Z)
                gen_y   = np.zeros((nbatch, 1))
                real_y  = np.ones((nbatch, 1))
                d_loss = self.dis_trainner.train_on_batch([gen_img, real_img], [gen_y, real_y])
                d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc = d_loss
                print '\td_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc'
                print '\t%.4f %.4f %.4f %.4f %.4f'%(d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc)

            if iteration%save_iter== 0:
                samples = self.generate(sample_zmb)
                vis_grid(samples, (5, 12), '{}/{}.png'.format(save_dir, iteration))
#               gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
#               dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)


from .losses import multiple_loss, mean_loss
from .layers import Subtract, GradNorm

class iWGAN(GAN):
    '''
        https://arxiv.org/pdf/1704.00028.pdf
    '''
    def generate(self, inputs):
        return self.generator.predict(inputs)

    def build(self, **kwargs):
        opt = kwargs['opt']
        lmbd = kwargs.get('lmbd', 10.0)

        gen, dis, gendis = self.generator, self.discriminator, self.gan

        dis.trainable = False 
        gendis.compile(optimizer=opt, loss=multiple_loss) # output: D(G(Z)) ===(-1*ones)===>  Loss:(-1) * D(G(Z)) 

        shape = dis.get_input_shape_at(0)[1:]
        gen_input, real_input, interpolation = Input(shape), Input(shape), Input(shape)
        sub = Subtract()([dis(gen_input), dis(real_input)])
        norm = GradNorm()([dis(interpolation), interpolation])
        dis2batch = Model([gen_input, real_input, interpolation], [sub, norm]) 
                            # output: D(G(Z))-D(X), norm ===(nones, ones)==> Loss: D(G(Z))-D(X)+lmbd*(norm-1)**2
        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss=[mean_loss,'mse'], loss_weights=[1.0, lmbd])


        self.gen_trainner = gendis
        self.dis_trainner = dis2batch

    def fit(self, data_generator, 
                niter=1000,
                nbatch=128,
                k=2,
                opt=None,
                save_dir='./quickshots/',
                save_iter=5):
        if opt == None: opt = Adam(lr=0.0001, beta_1=0.5, beta_2=0.9)
        self.build(opt=opt)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        vis_grid(data_generator(60), (5, 12), '{}/sample_real.png'.format(save_dir))
        sample_zmb = np.random.uniform(-1., 1., size=(60, self.coding)).astype('float32')
        vis_grid(self.generate(sample_zmb), (5, 12), '{}/sample_generate.png'.format(save_dir))

        g_loss, d_loss = 0, 0
        for iteration in range(1, niter+1):
            print 'iteration', iteration
            real_img = data_generator(nbatch)
            Z = np.random.uniform(-1., 1., size=(nbatch, self.coding)).astype('float32')

            if (k>0 and iteration%(k+1) == 0) or (k<0 and iteration%(-k+1) != 0): 
                y = np.ones((nbatch, 1)) * (-1)
                g_loss = self.gen_trainner.train_on_batch(Z, y) # output: D(G(Z)) ===(-1*ones)===>  Loss:(-1) * D(G(Z)) 
                g_loss = float(g_loss)
                print '\tg_loss=%.4f'%(g_loss)
            else:
                gen_img = self.generate(Z)
                epsilon = np.random.uniform(0, 1, size=(nbatch,1,1,1))
                interpolation = epsilon*real_img + (1-epsilon)*gen_img

                d_loss, d_diff, d_norm = self.dis_trainner.train_on_batch([gen_img, real_img, interpolation], [np.ones((nbatch, 1))]*2)
                print '\td_loss, d_diff, d_norm'
                print '\t%.4f %.4f %.4f'%(d_loss, d_diff, d_norm)

            if iteration%save_iter== 0:
                samples = self.generate(sample_zmb)
                vis_grid(samples, (5, 12), '{}/{}.png'.format(save_dir, iteration))
#               gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
#               dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)



