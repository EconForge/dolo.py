import numpy

def eval_with_diff(f, args, add_args, epsilon=1e-8):

    # f is a guvectorized function: f(x1, x2, ,xn, y1,..yp)
    # args is a list of vectors [x1,...,xn]
    # add_args is a list of vectors [y1,...,yn]
    # the function returns a list [r, dx1, ..., dxn] where:
    # r is the vector value value of f at (x1, xn, y1, yp)
    # dxi is jacobian w.r.t. xi

    # TODO: generalize when x1, ..., xn have non-core dimensions

    epsilon = 1e-8
    vec = numpy.concatenate(args)
    N = len(vec)
    points = vec[None,:].repeat(N+1, axis=0)
    for i in range(N):
        points[1+i,i] += epsilon

    argdims = [len(e) for e in args]
    cn = numpy.cumsum(argdims)
    slices = [e for e in zip( [0] + cn[:-1].tolist(), cn.tolist() )]
    vec_args = tuple([points[:,slice(*sl)] for sl in slices])

    arg_list = vec_args + add_args
    jac = f( *arg_list )
    res = jac[0,:]
    jac[1:,:] -= res[None,:]
    jac[1:,:] /= epsilon
    jacs = [jac[slice(sl[0]+1, sl[1]+1),:] for sl in slices]
    jacs = [j.T.copy() for j in jacs]  # to get C order
    return [res]  + jacs

class standard_function:

    epsilon = 1e-8

    def __init__(self, fun, n_output):

        # fun is a vectorized, non-allocating function
        self.fun = fun
        self.n_output = n_output

    def __call__(self, *args, diff=False, out=None):

        non_core_dims = [ a.shape[:-1] for a in args]
        core_dims = [a.shape[-1:] for a in args]

        non_core_ndims = [len(e) for e in non_core_dims]

        if (max(non_core_ndims) == 0):
            # we have only vectors, deal wwith it directly
            if not diff:
                if out is None:
                     out = numpy.zeros(self.n_output)
                self.fun(*(args+(out,)))
                return out

            else:
                def ff(*aa):
                    return self.__call__(*aa, diff=False)
                n_ignore = 1 # number of arguments that we don't differentiate
                res = eval_with_diff(ff, args[:-n_ignore], args[-n_ignore:], epsilon=1e-8)
                return res


        else:

            if not diff:
                K = max( non_core_ndims )
                ind = non_core_ndims.index( K )
                biggest_non_core_dim = non_core_dims[ind]
                biggest_non_core_dims = non_core_ndims[ind]
                new_args = []
                for i,arg in enumerate(args):
                    coredim = non_core_dims[i]
                    n_None = K-len(coredim)
                    n_Ellipsis = arg.ndim
                    newind = ((None,)*n_None) +(slice(None,None,None),)*n_Ellipsis
                    new_args.append(arg[newind])

                new_args = tuple(new_args)
                if out is None:
                    out = numpy.zeros( biggest_non_core_dim + (self.n_output,) )

                self.fun(*(new_args + (out,)))
                return out

            else:
                # older implementation
                return self.__vecdiff__(*args, diff=True, out=out)

    def __vecdiff__(self,*args, diff=False, out=None):


        fun = self.fun
        epsilon = self.epsilon

        sizes = [e.shape[0] for e in args if e.ndim==2]
        assert(len(set(sizes))==1)
        N = sizes[0]

        if out is None:
            out = numpy.zeros((N,self.n_output))

        fun( *( list(args) + [out] ) )

        if not diff:
            return out
        else:
            l_dout = []
            for i, a in enumerate(args[:-1]):
                # TODO: by default, we don't diffferentiate w.r.t. the last
                # argument. Reconsider.
                pargs = list(args)
                dout = numpy.zeros((N, self.n_output, a.shape[1]))
                for j in range( a.shape[1] ):
                    xx = a.copy()
                    xx[:,j] += epsilon
                    pargs[i] = xx
                    fun(*( list(pargs) + [dout[:,:,j]]))
                    dout[:,:,j] -= out
                    dout[:,:,j] /= epsilon
                l_dout.append(dout)
            return [out] + l_dout

def test_vector():

    def fun(s,x,p,out):
        out[0] = s[0] + x[0]
        out[1] = s[1] + x[1]

    s = numpy.random.random((2,))
    x = numpy.random.random((2,))
    p = numpy.random.random((2,))
    out = numpy.zeros((2,))
    out1 = numpy.zeros((2,))



    from numba import guvectorize, float64, void
    gfun = guvectorize(ftylist=[void(float64[:],float64[:],float64[:],float64[:])],
                        signature='(n),(n),(n)->(n)')(fun)

    sfun = standard_function(gfun,2)


    fun(s,x,p,out)
    sfun(s,x,p,out=out1)

    out2 = sfun(s,x,p)

    out = sfun(s,x,p,diff=True)

    print("OUT")
    print(out)



def test_columns():

    def fun(s,x,out):
        out[0] = s[0] + x[0]
        out[1] = s[1] + x[1]

    from numba import guvectorize, float64, void
    gfun = guvectorize(ftylist=[void(float64[:],float64[:],float64[:])],
                        signature='(n),(n)->(n)')(fun)


    N = 5

    s = numpy.random.random((N,2,))
    x = numpy.random.random((2,))

    out = numpy.zeros((N,2,))
    out1 = numpy.zeros((N,2,))

    sfun = standard_function(gfun,2)

    for n in range(N):
        fun(s[n,:],x,out[n,:])

    sfun(s,x,out=out1)

    out2 = sfun(s,x)



    # print(out2)
    # print(s+x)
    print(s+x)
    # assert( (abs(out2-s-x).max())<1e-8 )
    print(out)
    print(out1)
    print(out2)

    out, out_s,  = sfun(s,x,diff=True)


if __name__ == "__main__":
    test_vector()
    test_columns()
