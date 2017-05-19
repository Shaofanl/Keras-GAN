from GAN.utils.dist import *

if __name__ == '__main__':
    d = ProductDist([CategoryDist(3), UniformDist(-1.0, +1.0)])
    res = d.sample(10, ordered=True)

    print res

    import ipdb
    ipdb.set_trace()
