from keras.models import Model 
from keras.layers import Input, Dense, BatchNormalization, Activation, Deconv2D, Reshape

def basic_gen(input_shape, img_shape, nf=128, scale=4, FC=[]):
    dim, h, w = img_shape 

    img = Input(input_shape)
    x = img
    for fc_dim in FC: 
        x = Dense(fc_dim)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = Dense(nf*2**(scale-1)*(h/2**scale)*(w/2**scale))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Reshape((nf*2**(scale-1), h/2**scale, w/2**scale))(x)

    for s in range(scale-2, -1, -1):
        x = Deconv2D(nf*2**s, (3, 3), strides=(2, 2), padding='same')(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = Deconv2D(dim, (3, 3), strides=(2, 2), padding='same')(x) 
    x = Activation('tanh')(x)

    return Model(img, x)

