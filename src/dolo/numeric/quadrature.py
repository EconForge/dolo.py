import pytave
pytave.addpath('/home/pablo/Programmation/compecon/CEtools')
import numpy

def qnwnorm(orders, mu, sigma):

    orders = numpy.array(orders,dtype=float)
    mu = numpy.array(mu,dtype=float)
    sigma = numpy.array(sigma)

    [x,w] = pytave.feval(2,'qnwnorm',orders, mu, sigma)

    w = numpy.ascontiguousarray(w.flatten())
    x = numpy.ascontiguousarray(x.T)

    return [x,w]

if __name__ == '__main__':

    orders = [5,5]
    mu = [0.0,0.0]
    sigma = numpy.diag([0.1,0.1])

    [w,x] = qnwnorm(orders, mu, sigma)

    print w
    print x

    from matplotlib.pyplot import *

    plot(w[0,:],w[1,:],'o')
    show()