import numpy as np

def tensor(*args):
    return args

def rouwenhorst(rho=None, sigma=None, N=None):
    from dolo.numeric.discretization import rouwenhorst
    return rouwenhorst(rho,sigma,N)




class NumericEval:

    def __init__(self, d):

        self.d = d # dictionary of substitutions
        for k,v in d.iteritems():
            assert(isinstance(k, str))

        self.__supported_functions___ = [tensor, rouwenhorst]
        self.__supported_functions_names___ = [fun.__name__ for fun in self.__supported_functions___]

    def __call__(self, s):

        return self. eval(s)

    def eval(self, struct):

        t = struct.__class__.__name__
        method_name = 'eval_' + t.lower()
        try:
            fun = getattr(self, method_name)
        except Exception as e:
            raise Exception("Unknown type {}".format(method_name))
        return fun(struct)

    def eval_float(self, s):

        return s

    def eval_int(self, s):

        return s

    def eval_str(self, s):

        # expressions are not supported for now
        return self.d[s]


        # return eval(s, self.d)

    def eval_list(self, l):

        return [self.eval(e) for e in l]

    def eval_dict(self, d):

        if len(d) == 1:
            k = d.keys()[0]
            print('key: '+ k)
            if k in self.__supported_functions_names___:
                print(k)
                i = self.__supported_functions_names___.index(k)
                fun = self.__supported_functions___[i]

                args = d[k]
                if isinstance(args, dict):
                    eargs = self.eval(args)
                    res = fun(**eargs)
                elif isinstance(args, list):
                    eargs = self.eval(args)
                    res = fun(*eargs)
                return res


        return {k: self.eval(e) for k,e in d.iteritems()}

    def eval_ordereddict(self, s):

        from collections import OrderedDict
        res = OrderedDict()
        for k in s.keys():
            v = self.eval(s[k])
            res[k] = v

        return res

    def eval_ndarray(self, array_in):
        import numpy
        array_out = numpy.zeros_like(array_in, dtype=float)
        for i in range(array_in.shape[0]):
            for j in range(array_in.shape[1]):
                array_out[i,j] = self.eval(str(array_in[i,j]))
        return array_out

    def eval_nonetype(self, none):
        return None


# Markov mini language

if __name__ == '__main__':

    import numpy

    from collections import OrderedDict
    options = OrderedDict(
        smin= ['x',0.0],
        smax= ['y','x'],
        orders= [40,40],
        markov=dict(a=12.0, b=0.9)
    )


    d = {'x': 0.01, 'y': 10.0}
    print( NumericEval(d)(options) )



        # define a markov chain in yaml
    txt = '''
tensor:

    - rouwenhorst:
        rho: 0.9
        sigma: 0.4
        N: 3

    - markov:
        a: 0.8
        b: 1.2




    '''
    import yaml
    s = yaml.safe_load(txt)

    print(NumericEval(d)(s))




