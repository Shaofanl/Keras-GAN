import keras.backend as K

def fake_generate_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    return -K.log(y_pred).sum()

def cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0-K.epsilon())
    return -(y_true*K.log(y_pred) + (1-y_true)*K.log(1-y_pred)).sum()

   
