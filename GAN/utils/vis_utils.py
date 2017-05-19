import numpy as np
from scipy.misc import imsave

def vis_grid(X, (nh, nw), save_path=None):
    if X.shape[1] in [1, 3, 4]:
        X = X.transpose(0, 2, 3, 1)

    h, w, c = X.shape[1:]
    img = np.zeros((h*nh, w*nw, c))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        if n >= nh*nw: break
        img[j*h:j*h+h, i*w:i*w+w, :] = x

    if c == 1:
        img = img[:, :, 0]

    if save_path is not None:
        imsave(save_path, img)
    else:
        return img


