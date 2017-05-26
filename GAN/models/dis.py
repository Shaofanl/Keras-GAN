from keras.models import Model 
from keras.layers import Input, Dense, BatchNormalization, \
                         Activation, Conv2D, LeakyReLU, Flatten

def basic_dis(input_shape, nf=128, scale=4, FC=[], bn=True):
    dim, h, w = input_shape

    img = Input(input_shape)
    x = img
    
    for s in range(scale):
        x = Conv2D(nf*2**s, (5, 5), strides=(2, 2), padding='same')(x)
        if bn: x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)

    x = Flatten()(x)
    for fc in FC:
        x = Dense(fc)(x)
        if bn: x = BatchNormalization()(x)
        x = LeakyReLU(0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(img, x)

