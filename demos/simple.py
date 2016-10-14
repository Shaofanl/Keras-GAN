import numpy as np

from GAN.models import Generator, Discriminator, GAN

if __name__ == '__main__':
    g = Generator(g_size=(3, 32, 32), g_nb_filters=16, g_nb_noise=(50))
    d = Discriminator(d_size=g.g_size, d_nb_filters=16) 
    gan = GAN(g, d)

    x = g.random_generate()
    print x.shape
    print d.predict(x).shape

    print g, d, gan
