import numpy as np
import keras.backend as K

def InitNormal(shape, name=None):
    value = np.random.normal(loc=0.0, scale=0.02, size=shape)
    return K.variable(value, name=name)

