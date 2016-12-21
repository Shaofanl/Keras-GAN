import os
import numpy as np
np.set_printoptions(precision=3, suppress=True) 
import sys

from Generator import Generator
from Discriminator import Discriminator

import keras.backend as K
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input, Dense, Activation, Lambda

from ..utils.data import floatX, transform, inverse_transform
from ..utils.vis import vis_grid
from ..utils.loss import fake_generate_loss, cross_entropy_loss, masked_loss

from tqdm import tqdm 

_SAVE_ITER = 20

class GAN(object):
    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.gan = Sequential([generator, discriminator])
        self.generator = generator
        self.discriminator = discriminator

    def fit(self, data_stream, 
                nvis=120, 
                niter=1000,
                nbatch=128,
                nmax=None,
                k=2,
                opt=None,
                save_dir='./',
                transform=transform,
                inverse_transform=inverse_transform):
        if nmax is None: nmax = nbatch*100
        if opt == None: opt = Adam(lr=0.0001)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        gen, dis, gendis = self.generator, self.discriminator, self.gan
        gen_input, real_input = Input(dis.input_shape[1:]), Input(dis.input_shape[1:])
        dis2batch = Model([gen_input, real_input], [dis(gen_input), dis(real_input)])

        import theano
        theano.printing.pydotprint(gendis.outputs[0], outfile="pydotprint.pdf", format='pdf') 
        print 'debug saved'

        vis_grid(inverse_transform(transform(data_stream().next())), (5, 12), '{}/sample.png'.format(save_dir))
        sample_zmb = floatX(np.random.uniform(-1., 1., size=(nvis, gen.g_nb_coding)))
        vis_grid(inverse_transform(gen.generate(sample_zmb)), (5, 12), '{}/sample_generate.png'.format(save_dir))


        dis.trainable = False # must prevent dis from updating
        gendis.compile(optimizer=opt, loss='binary_crossentropy')#fake_generate_loss) # same effect when y===1
        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss='binary_crossentropy', 
                    metrics=['binary_accuracy'])#cross_entropy_loss)
#       dis.compile(optimizer=opt, loss='binary_crossentropy')#cross_entropy_loss)

        for iteration in range(niter):
            samples = gen.generate(sample_zmb)
            vis_grid(inverse_transform(samples), (5, 12), '{}/{}.png'.format(save_dir, iteration))

            ccc = 0
            n_updates = 0
            for real_img in data_stream(): #tqdm(data_stream(), total=nmax/nbatch):
                ccc += 1
                if ccc > nmax/nbatch: break

                Z = floatX(np.random.uniform(-1., 1., size=(nbatch, gen.g_nb_coding)))
                if (k>0 and n_updates % (k+1) == 0) or (k<0 and n_updates % (-k+1) != 0): 
                    y = np.array([[1] * nbatch]).reshape(-1, 1)
                    
                    g_loss = gendis.train_on_batch(Z, y)
                    g_loss = float(g_loss)
#                   print 'g_loss', g_loss
#                   vis_grid(inverse_transform(gen.predict(Z)), (10, 10), '{}/{}.g_train.png'.format(save_dir, iteration))
                else:
                    gen_img = gen.predict(Z)
                    real_img = transform(real_img)
                    
                    gen_y, real_y = np.zeros((nbatch, 1)), 
#                                   np.ones((nbatch, 1))
                                    np.random.binomial(1, 0.8, size=(100,)) # can boost training


#                   d_loss = dis.train_on_batch(gen_img, gen_y) + dis.train_on_batch(real_img, real_y) 
#                   d_loss = float(d_loss)
                    d_loss = dis2batch.train_on_batch([gen_img, real_img], [gen_y, real_y])
                    d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc = d_loss
#                   print 'd_loss', d_loss
#                   vis_grid(inverse_transform(np.concatenate((real_img[:50], gen_img[:50]))), (10, 10), '{}/{}.d_train.png'.format(save_dir, iteration))



                    # batch normalization:
                    #   if all inputs are from generated data, or all inputs are from real data, the performance is extremly poor
                    #   if the inputs are mixed with generated and real data, the performance is best
                n_updates += 1
            print 'n_epochs=%.0f, g_loss=%.4f, d_loss=%.4f\n'%(iteration, g_loss, d_loss)
            print 'd_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc'
            print '>> %.4f %.4f %.4f %.4f %.4f'%(d_loss, d_gen_loss, d_real_loss, d_gen_acc, d_real_acc)
            sys.stdout.flush()

            if iteration%_SAVE_ITER == 0:
                gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
                dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)

# end of GAN




#### previous version
#               Z = floatX(np.random.uniform(-1., 1., size=(nbatch, gen.g_nb_coding)))
#               if n_updates % (k+1) == 0:
#                   y = np.array([[1] * nbatch]).reshape(-1, 1)
#                   
#                   #print 'before', dis.predict(samples).sum() 
#                   g_loss = gendis.train_on_batch(Z, y)
#                   #print 'after', dis.predict(samples).sum() 
#                   g_loss = float(g_loss)
#                   print 'g_loss:', g_loss, 'acc: {}/{}'.format((gendis.predict(Z).round() == y).sum(), nbatch), 'mean', gendis.predict(Z).mean()
#                   vis_grid(inverse_transform(gen.predict(Z)), (10, 10), '{}/{}.g_train.png'.format(save_dir, iteration))
#               else:
#                   gen_img = gen.predict(Z)
#                   real_img = transform(real_img)

#                   X = np.concatenate((real_img, gen_img))
#                   y = np.array([[1] * nbatch + [0] *nbatch]).reshape(-1, 1)
#                   # keep the batch size <-- deprecated, case error because batchnormalization simplify the judgement
#                  #ind = np.random.choice(nbatch*2, size=nbatch)
#                  #X = X[ind]
#                  #y = y[ind]

#                   print 'before', dis.predict(gen_img).mean()
#                   print 'before', dis.predict(X[y.flatten()==0]).mean()
#                   d_loss = dis.train_on_batch(X[:nbatch], y[:nbatch]) + dis.train_on_batch(X[nbatch:], y[nbatch:])
#                   d_loss = float(d_loss)
#                   print 'after ', dis.predict(gen_img).mean()
#                   print 'after ', dis.predict(X[y.flatten()==0]).mean()

#                   pred = dis.predict(X[y.flatten()==0])
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == 0).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten()
#                   pred = dis.predict(X[y.flatten()==1])
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == 1).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten()
#                   pred = dis.predict(X)
#                   print 'd_loss:', d_loss, 'acc: {}/{}'.format((pred.round() == y).sum(), nbatch), 'mean', pred.mean(), 'pred', pred.flatten() 
#                   
#                   # batch normalization:
#                   #   if all inputs are from generated data, or all inputs are from real data, the performance is extremly poor
#                   #   if the inputs are mixed with generated and real data, the performance is best

#                   vis_grid(inverse_transform(np.concatenate((real_img[:50], gen_img[:50]))), (10, 10), '{}/{}.d_train.png'.format(save_dir, iteration))
#




class InfoGAN(object):
    def __init__(self, generator, discriminator, Qdist, **kwargs):
        super(InfoGAN, self).__init__(**kwargs)

#       self.gan = Sequential([generator, discriminator])
        self.Qdist = Qdist
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

        gen, dis, Q = self.generator, self.discriminator, self.Qdist
        dis_hidden = Sequential([layer for layer in dis.layers[:-1]])
        gen_input, real_input = Input(dis.input_shape[1:]), Input(dis.input_shape[1:])
        gen_coding = Input(gen.input_shape[1:])

        dis2batch = Model([gen_input, real_input], [dis(gen_input), dis(real_input)])
        neg_log_Q_c_given_x = -gen.g_info.register( # register and get log_Q_c_given_x
                                  gen.get_info_coding(gen_coding), # get oriented coding
                                  Q(dis_hidden(gen(gen_coding)))   # get parameters from Q network
                             )

        f = K.function([gen_coding], [neg_log_Q_c_given_x])
        res = f([gen.sample()])
        print res[0].shape

        neg_log_Q_c_given_x = K.sum(neg_log_Q_c_given_x, axis=1)
        neg_log_Q_c_given_x._keras_history = (Q, len(Q.inbound_nodes)-1, 0) # previous_layer, node_index, tensor_index
        neg_log_Q_c_given_x._uses_learning_phase = 0
        neg_log_Q_c_given_x._keras_shape = (None, 1)


        gendisQ = Model([gen_coding], [ dis(gen(gen_coding)), neg_log_Q_c_given_x ])

        dis.trainable = False # must prevent dis from updating
        gendisQ.compile(optimizer=opt, loss=['binary_crossentropy', masked_loss])#fake_generate_loss) # same effect when y===1

        dis.trainable = True # train dis model with 2 SEPARATE batch
        dis2batch.compile(optimizer=opt, loss='binary_crossentropy')#cross_entropy_loss)

        vis_grid(data_stream().next(), (12, 10), '{}/sample.png'.format(save_dir))
        sample_zmb = floatX(gen.sample(120, ordered=True, orderedN=10))

        for iteration in range(niter):
            samples = gen.predict(sample_zmb)
            vis_grid(inverse_transform(samples), (12, 10), '{}/{}.png'.format(save_dir, iteration))

            ccc = 0
            n_updates = 0
            for real_img in tqdm(data_stream(), total=nmax/nbatch):
                ccc += 1
                if ccc > nmax/nbatch: break

                Z = floatX(gen.sample(nbatch))
                if n_updates % (k+1) == 0:
                    y = np.array([[1] * nbatch]).reshape(-1, 1)
                    
                    g_loss_sum, gen_loss, Q_loss = gendisQ.train_on_batch([Z], [y, y])
                    g_loss_sum  = float(g_loss_sum)
                    gen_loss    = float(gen_loss)
                    Q_loss      = float(Q_loss)
                else:
                    gen_img = gen.predict(Z)
                    real_img = transform(real_img)

                    gen_y, real_y = np.zeros((nbatch, 1)), np.ones((nbatch, 1))

#                   d_loss = dis.train_on_batch(gen_img, gen_y) + dis.train_on_batch(real_img, real_y) 
#                   d_loss = float(d_loss)
                    d_loss = dis2batch.train_on_batch([gen_img, real_img], [gen_y, real_y])
                    d_loss = float(d_loss[0])

                    # batch normalization:
                    #   if all inputs are from generated data, or all inputs are from real data, the performance is extremly poor
                    #   if the inputs are mixed with generated and real data, the performance is best
                n_updates += 1
            print 'n_epochs=%.0f, g_loss=%.4f=%.4f+%.4f, d_loss=%.4f\n'%(iteration, g_loss_sum, gen_loss, Q_loss, d_loss)

            if iteration% _SAVE_ITER== 0:
                gen.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
                dis.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)
# end of InfoGAN


class AEGAN(object):
    def __init__(self, generator, discriminator, autoencoder, **kwargs):
        super(AEGAN, self).__init__(**kwargs)

        self.gan = Sequential([generator, discriminator])
        self.generator = generator
        self.discriminator = discriminator
        self.autoencoder = autoencoder
#       self.encoder = Sequential([layer for layer in discriminator.layers[:-1]]+\
#                                 [Dense(generator.g_nb_coding, activation='tanh')])
#       self.ae = Sequential([layer for layer in self.encoder.layers]+[layer for layer in generator.layers])

    def load(self, prefix):
        self.generator.load_weights(prefix+'_gen_params.h5')
        self.discriminator.load_weights(prefix+'_dis_params.h5')
        self.autoencoder.autoencoder.load_weights(prefix+'_ae_params.h5')
        print 'loading done'

    def encode(self, inputs, nbatch=128):
        return self.autoencoder.encoder.predict(inputs, batch_size=nbatch)

    def fit(self, data_stream, 
                nvis=120, 
                niter=1000,
                nbatch=128,
                nmax=None,
                k=2,
                opt=None,
                pull=False, pullx=None, pully=None, pullcoef=None,
                rec_with_only_dis=True,
                save_dir='./'):
        if nmax is None: nmax = nbatch*100
        if opt == None: opt = Adam(lr=0.0001)
        if not os.path.exists(save_dir): os.makedirs(save_dir)

        gen, dis = self.generator, self.discriminator 
        # fix dis and train gen
        gendis = self.gan
        # train on two batch separately but update simultaneously
        gen_input, real_input = Input(dis.input_shape[1:]), Input(dis.input_shape[1:])
        dis2batch = Model([gen_input, real_input], [dis(gen_input), dis(real_input)])
        # autoencoder to reconstruct 
        if pull:
            input_a = Input(self.autoencoder.encoder.input_shape[1:])
            input_b = Input(self.autoencoder.encoder.input_shape[1:])
            code_a = self.autoencoder.encoder(input_a)
            code_b = self.autoencoder.encoder(input_b)
            def euclidean_distance(vects):
                x, y = vects
                return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))
            def eucl_dist_output_shape(shapes):
                shape1, shape2 = shapes
                return (shape1[0], 1)
            distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([code_a, code_b])

            ae = Model([self.autoencoder.autoencoder.input, input_a, input_b], [self.autoencoder.autoencoder.output, distance])
#           modelA = self.autoencoder.autoencoder
#           modelB = Model([input_a, input_b], [distance])
        else:
            ae = self.autoencoder.autoencoder


        import theano
        theano.printing.pydotprint(gendis.outputs[0], outfile="pydotprint.pdf", format='pdf')
        print 'debug saved'

        vis_grid(data_stream().next(), (5, 12), '{}/sample.png'.format(save_dir))
        sample_zmb = floatX(np.random.uniform(-1., 1., size=(nvis, gen.g_nb_coding)))
        vis_grid(inverse_transform(gen.generate(sample_zmb)), (5, 12), '{}/sample_generate.png'.format(save_dir))


        # train generative network
        dis.trainable = False # must prevent dis from updating
        gendis.compile(optimizer=opt, loss='binary_crossentropy')#fake_generate_loss) # same effect when y===1
        # train discriminative network
        dis.trainable = True
        dis2batch.compile(optimizer=opt, loss='binary_crossentropy')#cross_entropy_loss)
#       dis.compile(optimizer=opt, loss='binary_crossentropy')#cross_entropy_loss)
        # jointly trained for reconstruction
        if rec_with_only_dis:
            for layer in ae.layers:
                if layer in gen.layers:
                    layer.trainable = False
        else:
            pass

        if pull:
            def contrastive_loss(y_true, y_pred):
                margin = 1.0
                return pullcoef*K.mean(y_true * K.square(y_pred))# + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))
            ae.compile(optimizer=opt, loss=['mse', contrastive_loss])
#           modelA.compile(optimizer=opt, loss='mse')
#           modelB.compile(optimizer=opt, loss=contrastive_loss)

#           _, rec_loss, pull_loss = ae.train_on_batch([real_img], [real_img])
#           pull_ind = np.random.choice(len(pullx)/2, size=(nbatch, ))
#           input_a = pullx[pull_ind]
#           input_b = pullx[pull_ind+len(pullx)/2]
#           real_img = data_stream().next()
#           real_img = transform(real_img)
#           _, rec_loss, pull_loss = ae.train_on_batch([real_img, input_a, input_b], [real_img, np.ones((nbatch, 1))])
#           print 'test done'
        else:
            ae.compile(optimizer=opt, loss='mse')




        for iteration in range(niter):
            samples = gen.generate(sample_zmb)
            vis_grid(inverse_transform(samples), (5, 12), '{}/{}.png'.format(save_dir, iteration))

            ccc = 0
            n_updates = 0
            sys.stdout.flush()
            for real_img in tqdm(data_stream(), total=nmax/nbatch):
                ccc += 1
                if ccc > nmax/nbatch: break

                Z = floatX(np.random.uniform(-1., 1., size=(nbatch, gen.g_nb_coding)))
                gen_img = gen.predict(Z)
                real_img = transform(real_img)
                if (k>0 and n_updates % (k+1) == 0) or (k<0 and n_updates % (-k+1) != 0): 
                    y = np.array([[1] * nbatch]).reshape(-1, 1)
                    
#                   if np.isnan(gendis.evaluate(Z, y)).any():
#                       import ipdb
#                       ipdb.set_trace()
                    g_loss = gendis.train_on_batch(Z, y)
                    g_loss = float(g_loss)
                    print 'g_loss', g_loss
#                   vis_grid(inverse_transform(gen.predict(Z)), (10, 10), '{}/{}.g_train.png'.format(save_dir, iteration))
                else: 
                    gen_y, real_y = np.zeros((nbatch, 1)), np.ones((nbatch, 1))

#                   d_loss = dis.train_on_batch(gen_img, gen_y) + dis.train_on_batch(real_img, real_y) 
#                   d_loss = float(d_loss)
#                   if np.isnan(dis2batch.evaluate([gen_img, real_img], [gen_y, real_y])).any():
#                       import ipdb
#                       ipdb.set_trace()
                    d_loss = dis2batch.train_on_batch([gen_img, real_img], [gen_y, real_y])
                    d_loss = float(d_loss[0])
                    print 'd_loss', d_loss
#                   vis_grid(inverse_transform(np.concatenate((real_img[:50], gen_img[:50]))), (10, 10), '{}/{}.d_train.png'.format(save_dir, iteration))


                    # batch normalization:
                    #   if all inputs are from generated data, or all inputs are from real data, the performance is extremly poor
                    #   if the inputs are mixed with generated and real data, the performance is best
                if pull:
                    assert pullx != None 

#                   import ipdb
#                   ipdb.set_trace()
                    # supervised
                    pull_ind = np.random.choice(len(pullx)/2, size=(nbatch, ))
                    input_a = pullx[pull_ind]
                    input_b = pullx[pull_ind+len(pullx)/2]
#                   res = ae.predict([real_img, input_a, input_b], batch_size=128, verbose=0)
#                   if np.isnan(res[0]).any() or np.isnan(res[1]).any():
#                       import ipdb
#                       ipdb.set_trace()
                    code = self.autoencoder.encoder.predict(pullx)
#                   import ipdb
#                   ipdb.set_trace()
                    _, rec_loss, pull_loss = ae.train_on_batch([real_img, input_a, input_b], [real_img, np.ones((nbatch, 1))])
                    print 'rec_loss', rec_loss, 'pull_loss', pull_loss
#                   rec_loss = modelA.train_on_batch(real_img, real_img)
#                   print 'rec_loss', rec_loss
#                   pull_loss = modelB.train_on_batch([input_a, input_b], [np.ones((nbatch, 1))])
#                   print 'pull_loss', pull_loss
                else:
#                   import ipdb
#                   ipdb.set_trace()
                    rec_loss = ae.train_on_batch(real_img, real_img)
                    print 'rec_loss', rec_loss
                n_updates += 1
            if pull:
                print '\nn_epochs=%.0f, g_loss=%.4f, d_loss=%.4f, rec_loss=%.4f, pull_loss=%.4f\n'%(iteration, g_loss, d_loss, rec_loss, pull_loss)
            else:
                print '\nn_epochs=%.0f, g_loss=%.4f, d_loss=%.4f, rec_loss=%.4f\n'%(iteration, g_loss, d_loss, rec_loss)

            if iteration% _SAVE_ITER== 0:
                self.generator.save_weights('{}/{}_gen_params.h5'.format(save_dir, iteration), overwrite=True)
                self.discriminator.save_weights('{}/{}_dis_params.h5'.format(save_dir, iteration), overwrite=True)
                self.autoencoder.autoencoder.save_weights('{}/{}_ae_params.h5'.format(save_dir, iteration), overwrite=True)

# end of AEGAN




class CondGAN(object):
    pass
# end of CondGAN



