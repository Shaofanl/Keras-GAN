import numpy as np
import keras.backend as K

def InitNormal(loc=0.0, scale=0.002):
    def initf(model):
        for w in model.weights:
            if w.name.startswith('conv2d') or w.name.startswith('dense'):
                if 'kernel' in w.name:
                    print 'init weight', w.name
                    value = np.random.normal(loc=loc, scale=scale, size=K.get_value(w).shape)
                    K.set_value(w, value.astype('float32'))
    return initf

