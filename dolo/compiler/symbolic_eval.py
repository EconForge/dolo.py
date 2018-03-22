import numpy as np

from dolo.numeric.discretization import tensor_markov

# from dolo.compiler.language import minilang
from dolo.compiler.language import minilang, functions


class NumericEval:
    def __init__(self, d, minilang=minilang):

        self.d = d  # dictionary of substitutions
        for k, v in d.items():
            assert (isinstance(k, str))
        for k, v in functions.items():
            d[k] = v

        self.minilang = minilang

    def __call__(self, s):

        return self.eval(s)

    def eval(self, struct):

        tt = tuple(self.minilang)

        if isinstance(struct, tt):
            return struct.eval(self.d)

        t = struct.__class__.__name__

        method_name = 'eval_' + t.lower()
        try:
            fun = getattr(self, method_name)

        except Exception:
            raise Exception("Unknown type {}".format(method_name))

        return fun(struct)

    def eval_scalarfloat(self, s):
        return float(s)

    def eval_float(self, s):

        return s

    def eval_scalarfloat(self, s):

        return float(s)

    def eval_scalarfloat(self, s):

        return float(s)

    def eval_int(self, s):

        return s

    def eval_str(self, s):

        # not safe
        return eval(s, self.d)

    def eval_list(self, l):

        return [self.eval(e) for e in l]

    def eval_dict(self, d):

        return {k: self.eval(e) for k, e in d.items()}

    def eval_ordereddict(self, s):

        res = dict()
        for k in s.keys():
            v = self.eval(s[k])
            res[k] = v

        return res

    def eval_commentedseq(self, s):
        return self.eval_list(s)

    def eval_ndarray(self, array_in):
        import numpy
        array_out = numpy.zeros_like(array_in, dtype=float)
        for i in range(array_in.shape[0]):
            for j in range(array_in.shape[1]):
                array_out[i, j] = self.eval(array_in[i, j])
        return array_out

    def eval_nonetype(self, none):
        return None


# Markov mini language

if __name__ == '__main__':

    import numpy

    options = dict(
        smin=['x', 0.0],
        smax=['y', 'x'],
        orders=[40, 40],
        markov=dict(a=12.0, b=0.9))

    d = {'x': 0.01, 'y': 10.0}
    print(NumericEval(d)(options))

    # define a markov chain in yaml
    txt = '''
tensor:

    - rouwenhorst:
        rho: 0.9
        Sigma: 0.4
        N: 3

    - markov:
        a: 0.8
        b: 1.2




    '''
    import yaml
    s = yaml.safe_load(txt)

    print(NumericEval(d)(s))
