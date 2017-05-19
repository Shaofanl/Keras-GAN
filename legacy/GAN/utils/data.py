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


_random_color_mapping = np.random.random((8, 3))
_ncp_code = [0, 9, 19, 29, 39, 50, 60, 62] #   Hair = 9, Face = 19, UpperClothes = 29, LowerClothes = 39, Arms = 50, Legs = 60, Shoes = 62, and Background = 0.
def transform_skeleton(X):
    x = floatX(X).transpose(0, 3, 1, 2)
    assert x.shape[1] == 1

    newx = np.zeros(shape=(x.shape[0], 8, x.shape[2], x.shape[3]))
    for index, code in enumerate(_ncp_code):
        newx[:, index, :, :] = (x[:, 0, :, :] == code).astype('float32') * 255.
    x = floatX( newx )
    return x/127.5 - 1.

def inverse_transform_skeleton(X):
#   newx = np.zeros(shape=(X.shape[0], 1, npxh, npxw))
    X = (X+1.)/2.  # [-1, 1] --> [0, 1]
    newx = X.swapaxes(1,3).dot(_random_color_mapping).swapaxes(1,3)
    return floatX(newx.transpose(0, 2, 3, 1))


 
