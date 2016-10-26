import numpy as np
import keras.backend as K

def InitNormal(loc=0.0, scale=0.002):
    def initf(shape, name=None):
        value = np.random.normal(loc=loc, scale=scale, size=shape)
        return K.variable(value, name=name)
    return initf

