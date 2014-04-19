from numpy import array, zeros
from dolo.numeric.misc import mlinspace


class MarkovDecisionRule:


    def __init__(self, n_m, a, b, orders, values=None):

        dtype = numpy.double
        self.n_m = int(n_m)
        self.a = array(a, dtype=dtype)
        self.b = array(b, dtype=dtype)
        self.orders = array(orders, dtype=int)

        # for backward compatibility
        self.smin = self.a
        self.smax = self.b

        self.dtype = dtype

        self.N = self.orders.prod()

        self.__grid__ = None

        if values is not None:
            self.set_values(values)
        else:
            self.__values__ = None

    @property
    def grid(self):

        if self.__grid__ is None:
            self.__grid__ = mlinspace(self.smin, self.smax, self.orders)
        return self.__grid__

    def set_values(self, values):

        self.__values__ = values
        self.__coefs__ = filter(self.smin, self.smax, self.orders, values)

    def __call__(self, i_m, points, out=None):

        n_x = self.__values__.shape[-1]

        if points.ndim == 2:

            # each line is supposed to correspond to a new point
            N,d = points.shape
            assert(d==len(self.orders))

            out = zeros((N,n_x))
            for n in range(N):
                self.__call__(i_m, points[n,:], out[n,:])

            return out

        else:
            if out == None:
                out = zeros(n_x)
            coefs = self.__coefs__[i_m,...]

            # TODO: replace
            eval_UB_spline(self.a, self.b, self.orders, coefs, points, out)
            return out