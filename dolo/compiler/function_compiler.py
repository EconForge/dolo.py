import numpy

class standard_function:

    epsilon = 1e-8

    def __init__(self, fun, n_output):

        # fun is a vectorized, allocating function
        self.fun = fun
        self.n_output = n_output

    def __call__(self, *args, **kwargs):


        is_vector = (args[0].ndim == 1)

        if is_vector:

            nargs = [e[None,:] for e in args]
            # print(args)
            resp = self.__callvec__(*nargs, **kwargs)

            if 'diff' in kwargs:
                return [e[0,...] for e in resp]
            else:
                return resp[0,...]
        else:
            return self.__callvec__(*args, **kwargs)



    def __callvec__(self,*args, **kwargs):


        fun = self.fun
        epsilon = self.epsilon

        sizes = [e.shape[0] for e in args if e.ndim==2]
        assert(len(set(sizes))==1)
        N = sizes[0]

        # if 'out' in kwargs:
        #     out = kwargs['out']
        # else:

        out = numpy.zeros((N,self.n_output))

        fun( *args , out=out )

        # if ('out' not in kwargs):
        #     return out

        if not 'diff' in kwargs:
            
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
                    fun(*pargs, out=dout[:,:,j])
                    dout[:,:,j] -= out
                    dout[:,:,j] /= epsilon

                l_dout.append(dout)

            return [out] + l_dout
