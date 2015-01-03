from numpy import atleast_2d, dot

class TaylorExpansion:

    def __init__(self,*l):

        if len(l) == 1:
            # backward compatibility
            l = l[0]

        l = tuple(l)

        self.order = len(l) - 2
        self.coefs = l
        self.S_bar = l[0]
        self.X_bar = l[1]
        self.X_s = l[2]
        if self.order >= 2:
            self.X_ss = l[3]
        if self.order >= 3:
            self.X_sss = l[4]
        if self.order >=4:
            raise Exception("Not implemented")

    def __getitem__(self, ind):

        l = [ self.S_bar.copy() ]
        l.append(self.X_bar[ind].copy())
        l.append(self.X_s[ind,...].copy())
        if self.order >= 2:
            l.append(self.X_ss[ind,...].copy())
        if self.order >= 3:
            l.append(self.X_sss[ind,...].copy())
        return TaylorExpansion( l )


    def __call__(self, points):

        if self.order == 1:
            return eval_te_order_1(self.S_bar, self.X_bar, self.X_s, points)
        elif self.order == 2:
            return eval_te_order_2(self.S_bar, self.X_bar, self.X_s, self.X_ss, points)
        elif self.order == 3:
            return eval_te_order_3(self.S_bar, self.X_bar, self.X_s, self.X_ss, self.X_sss, points)

    def __call2__(self,points):

        from numpy import tile
        from dolo.numeric.tensor import mdot


        # slower implementation, kept for comparison purpose
        if points.ndim == 1:
            pp = atleast_2d(points)
            res = self.__call__(pp)
            return res.ravel()

        points = points.T

        n_s = points.shape[1]
        ds = points - self.S_bar[:,None]
        choice =  dot(self.X_s, ds) + self.X_bar[:,None]
        n_ss = self.X_s.shape[1]
        if self.order == 2:
            for k in range(self.X_ss.shape[0]):
                for i in range(n_ss):
                    for j in range(n_ss):
                        choice[k,:] += self.X_ss[k,i,j]*ds[i,:]*ds[j,:]/2
        if self.order == 3:
            for i in range(n_s):
                choice[:,i] += mdot(self.X_ss,[ds[:,i],ds[:,i]]) / 2
                choice[:,i] += mdot(self.X_sss,[ds[:,i],ds[:,i],ds[:,i]]) / 6

        return choice.T


# helper functions

from numba import guvectorize

@guvectorize(['void(f8[:],f8[:],f8[:,:],f8[:],f8[:])'], '(s),(x),(x,s),(s)->(x)')
def eval_te_order_1(s0, x0, x1, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n,i]*(points[i]-s0[i])

    # return out

@guvectorize(['void(f8[:],f8[:],f8[:,:],f8[:,:,:],f8[:],f8[:])'], '(s),(x),(x,s),(x,s,s),(s)->(x)')
def eval_te_order_2(s0, x0, x1, x2, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n,i]*(points[i]-s0[i])
            for j in range(S):
                out[n] += x2[n,i,j]*(points[i]-s0[i])*(points[j]-s0[j])/2.0

@guvectorize(['void(f8[:],f8[:],f8[:,:],f8[:,:,:], f8[:,:,:,:], f8[:],f8[:])'], '(s),(x),(x,s),(x,s,s),(x,s,s,s),(s)->(x)')
def eval_te_order_3(s0, x0, x1, x2, x3, points, out):

    S = points.shape[0]
    X = x0.shape[0]
    for n in range(X):
        out[n] = x0[n]
        for i in range(S):
            out[n] += x1[n,i]*(points[i]-s0[i])
            for j in range(S):
                out[n] += x2[n,i,j]*(points[i]-s0[i])*(points[j]-s0[j])/2.0
                for k in range(S):
                    out[n] += x3[n,i,j,k]*(points[i]-s0[i])*(points[j]-s0[j])*(points[k]-s0[k])/6.0





def test_taylor_expansion():

    import numpy
    from numpy import array

    s0 = array([0.2, 0.4, 1.1])
    x0 = array([1.2, 0.9])

    N = 1000
    points = numpy.random.random((N,3))

    X_s = numpy.random.random((2,3))
    X_ss = numpy.random.random((2,3,3))
    X_sss = numpy.random.random((2,3,3,3))
    dr1 = TaylorExpansion(s0,x0,X_s) #, X_ss, X_sss])
    dr2 = TaylorExpansion(s0,x0,X_s,X_ss) #, X_sss])
    dr3 = TaylorExpansion(s0,x0,X_s, X_ss,X_sss)

    out1 = dr1(points)
    out2 = dr2(points)
    out3 = dr3(points)

    out1_1d = dr1(points[0,:])
    out2_1d = dr2(points[0,:])
    out3_1d = dr3(points[0,:])

    assert(abs(out1_1d - out1[0,:]).max()==0)
    assert(abs(out2_1d - out2[0,:]).max()==0)
    assert(abs(out3_1d - out3[0,:]).max()==0)

    ds = points-s0[None,:]
    ds2 = ds[:,:,None].repeat(3, axis=2)
    ds3 = ds2[:,:,:,None].repeat(3, axis=3)


    from numpy import dot
    verif1 = x0[None,:] + numpy.dot(ds, X_s.T)
    verif2 = dr2.__call2__(points)
    verif3 = dr3.__call2__(points)

    assert(abs(out1-verif1).max()<1e-12)
    assert(abs(out2-verif2).max()<1e-12)
    assert(abs(out3-verif3).max()<1e-12)


if __name__ == "__main__":

    test_taylor_expansion()
