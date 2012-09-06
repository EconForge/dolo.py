import pytave
import numpy
import os


class CEInterpolation:
    def __init__(self,atype, smin, smax, orders):
        self.smin = smin
        self.smax = smax
        self.bounds = numpy.row_stack([smin,smax])
        self.d = len(smin)
        self.orders = orders
        self.__interp__ = pytave.feval(1, 'create_space', atype, orders, smin,smax) # we don't really need it here..
        self.grid = self.__interp__[0]['points'][0,0].T

    def set_values(self,values):
        pytave.feval(0,'set_coeffs',values.T)[0]

    def __call__(self, points):
        derivs = numpy.zeros((1,self.d))
        values = pytave.feval(1,'my_funeval',points.T,derivs)[0]
        return values.T

    def interpolate(self,points):
        derivs = numpy.row_stack( [numpy.zeros((1,self.d)), numpy.eye(self.d)] )
        resp = pytave.feval(1,'my_funeval',points.T,derivs)[0]
        val = resp[:,:,0]
        dval = resp[:,:,1:]
        val = val.T
        dval = numpy.rollaxis(dval,0,3)
        return [val,dval]



if __name__ == '__main__':


    pytave.feval(0,'addpath','/home/pablo/Programmation/compecon/CEtools')
    pytave.feval(0,'addpath','/home/pablo/Documents/Research/Thesis/Chapter_1/code/')

    orders = numpy.array( [ 5, 5  ] )
    smin  = numpy.array( [ 0.9,  0.5 ] )
    smax  = numpy.array( [ 1.1,  2 ] )

    f = lambda x: x[0:1,:] * x[1:2,:]


    interp = CEInterpolation('spli',smin,smax,orders)

    values = f(interp.grid)

    print values

    interp.set_values(values)

    print interp(numpy.column_stack( [interp.grid, interp.grid] ) )