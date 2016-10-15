import numpy as np
import theano

def floatX(x):
    return np.array(x).astype(theano.config.floatX)

def inverse_transform(X):
    X = X.transpose(0, 2, 3, 1)+1./2.
    return X

def transform(X):
# random_crop
#   X = [center_crop(x, npxh, npxw) for x in X]
    return X.transpose(0, 3, 1, 2)/127.5 - 1.

  
