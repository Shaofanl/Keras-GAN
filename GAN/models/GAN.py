from Generator import Generator
from Discriminator import Discriminator

from keras.models import Sequential, Model

class GAN(Sequential):
    def __init__(self, generator, discriminator, **kwargs):
        super(GAN, self).__init__(**kwargs)

        self.add(generator)
        self.add(discriminator)
        
        self.generator = generator
        self.discriminator = discriminator
