import numpy as np

class NumericEval:

    def __init__(self, d):

        self.d = d # dictionary of substitutions
        for k,v in d.iteritems():
            assert(isinstance(k, str))

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

        return eval(s, self.d)

    def eval_list(self, l):

        return [self.eval(e) for e in l]

    def eval_dict(self, d):

        return {k: self.eval(e) for k,e in d.iteritems()}

    def eval_ordereddict(self, s):

        from collections import OrderedDict
        res = OrderedDict()
        for k in s.keys():
            v = self.eval(s[k])
            res[k] = v

        return res


if __name__ == '__main__':

    import numpy

    from collections import OrderedDict
    options = OrderedDict(
        smin= ['x',0.0],
        smax= ['y','x+y'],
        orders= [40,40]
    )


    d = {'x': 0.01, 'y': 10.0}
    print( NumericEval(d)(options) )


