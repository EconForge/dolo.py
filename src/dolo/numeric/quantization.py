import numpy


quantization_data = '/home/pablo/Documents/Research/Thesis/Chapter_1/code/data/quantization_grids/'


def quantization_weights(N,sigma):
    # currently only the diagonal matters
    assert( len(sigma.shape) == 2 )
    assert ( abs( numpy.diag(numpy.diag(sigma)) - sigma).max() == 0 ) # TODO : here we don't allow for cross correlations.
    var = numpy.diag(sigma)
    sd = numpy.sqrt(var)
    sd = numpy.diag(sd)
    d = sigma.shape[0]
    [w, x] = standard_quantization_weights(N,d )
    x = numpy.dot(sd, x)
    return [w,x]

def standard_quantization_weights(N,d):
    filename = quantization_data + '{0}_{1}_nopti'.format(N,d)
    import numpy

    try:
        G = numpy.loadtxt(filename)
    except Exception as e:
        raise e

    w = G[:N,0]
    x = G[:N,1:d+1]
    return [w,x.T]

if __name__ == '__main__':
    N = 8
    d = 4
    import numpy
    sigma = numpy.diag([0.01,0.01,0.01,0.01])
    [w,x] = quantization_weights(N,sigma)
    print 'w'
    print w
    print 'x'
    print x
