from collections import OrderedDict

import math

import numpy

def filter(smin, smax, orders, controls):
    
    from dolo.numeric.interpolation.splines_filter import filter_data
    n_mc, N, n_x = controls.shape
    dinv = (smax-smin)/(orders-1)
    oorders = ( orders + 2 ).tolist()
    res = zeros( [n_mc] + [n_x] + oorders )
   
    for i in range(n_mc):
        for j in range(n_x):
            ddd = controls[i,:,j].reshape(orders).copy()
            rhs = filter_data(dinv, ddd) #, dinv )
            res[i,j,...] = rhs
    return res


def calibration_to_vector(symbols, calibration_dict):
  
    from triangular_solver import solve_triangular_system

    import time

    sol = solve_triangular_system(calibration_dict)
    
    calibration  = OrderedDict()
    for group in symbols:
        calibration[group] = numpy.array(
                                [sol[s] for s in symbols[group]],
                             dtype=float)
    
    return calibration


def calibration_to_dict(model, calib):

    from collections import OrderedDict

    d = OrderedDict()
    for group, values in calib.iteritems():
        if group == 'covariances':
            continue
        syms = model.symbols[group]
        for i,s in enumerate(syms):
            d[s] = values[i]

    return d




def allocating_function(inplace_function, size_output):

    import inspect

    def new_function(*args, **kwargs):
        val = numpy.zeros(size_output)
        nargs = args + (val,) 
        inplace_function( *nargs )
        if 'diff' in kwargs:
            return numdiff(new_function, args)
        return val

    return new_function


def numdiff(fun, args):

    # vectorized version

    epsilon = 1e-8
    args = list(args)
    v0 = fun(*args)
    N = v0.shape[0]
    l_v = len(v0)
    dvs = []
    for i,a in enumerate(args):
        l_a = (a).shape[1]
        dv = numpy.zeros( (N, l_v, l_a) )
        nargs = list(args) #.copy()
        for j in range(l_a):
            xx = args[i].copy()
            xx[:,j] += epsilon
            nargs[i] = xx
            dv[:,:,j] = (fun(*nargs) - v0)/epsilon
        dvs.append(dv)
    return [v0] + dvs

# def numdiff(fun, args):

#     epsilon = 1e-8
#     args = list(args)
#     v0 = fun(*args)
#     l_v = len(v0)
#     dvs = []
#     for i,a in enumerate(args):
#         l_a = len(a)
#         dv = numpy.zeros( (l_v, l_a) )
#         nargs = list(args) #.copy()
#         for j in range(l_a):
#             xx = args[i].copy()
#             xx[j] += epsilon
#             nargs[i] = xx
#             dv[:,j] = (fun(*nargs) - v0)/epsilon
#         dvs.append(dv)
#     return [v0] + dvs



def check(model, silent=False):

    from numpy import concatenate
    from numpy.testing import assert_almost_equal

    from collections import OrderedDict

    checks = OrderedDict()

#    for name in model.calibration.keys():
#        assert( [len(model.calibration[name])==len(model.symbols[name])] )

    names = ['markov_states', 'states', 'controls', 'parameters']

    m,s,x,p = [model.calibration[name] for name in names]
    
    
    # check steady_state
    from numba import jit
    g = model.functions['transition']
    S = g(m,s,x,m, p)
    

    R = model.functions['arbitrage'](m,s,x, m, s, x,p)
    e = abs( concatenate([S-s,R]) ).max()

    try:
        assert(e<1e-8)
        checks['steady_state'] = True
    except:
        checks['steady_state'] = False
        if not silent:
            print("Residuals :")
            print("Transitions :\n{}".format(S-s))
            print("Arbitrage :\n{}".format(R))
            raise Exception("Non zero residuals at the calibrated values.")


    if 'markov_nodes' in model.options:

        n_ms = len(model.symbols['markov_states'])

        P = model.options['markov_transitions']
        N = model.options['markov_nodes']
        
        try:
            assert( N.shape[1] == n_ms )
        except:
            raise Exception("Markov nodes incorrect. Should have {} columns instead of {}".format(n_ms, N.shape[1]))

        n_mc = P.shape[0]
        assert(P.shape[1] == n_mc)

        ss = P.sum(axis=1)
        for i in range(n_mc):
            try:
                q = ss[i]
                assert_almost_equal(q,1)
            except:
                raise Exception("Markov transitions incorrect. Row {} sums to {} instead of 1.".format(i,q))

    if 'approximation_space' in model.options:
        
        ap = model.options['approximation_space']
        smin = ap['smin']
        smax = ap['smax']
        orders = ap['orders']

        n_states = len(model.symbols['states'])

        assert(len(smin)==n_states)
        assert(len(smax)==n_states)
        assert(len(orders)==n_states)

        for i in range(n_states):
            try:
                assert(smin[i]<smax[i])
            except:
                raise Exception("Incorrect bounds for state {} ({}). Min is {}. Max is {}.".format(i,model.symbols['states'][i],smin[i],smax[i]))

            try:
                assert(int(orders[i])==orders[i])
            except:
                raise Exception("Incorrect number of nodes for state {} ({}). Found {}. Must be greater than or equal to {}".format(i,model.symbols['states'][i],orders[i]))

    return checks
       

def compute_residuals(model,calib=None):
    
    if calib is None:
        calib = model.calibration

    s = calib['states']
    x = calib['controls']
    p = calib['parameters']
    m = calib['markov_states']

    g = model.functions['transition']

    S = g(m,s,x,m,p)
    res = model.functions['arbitrage'](m,s,x,m,s,x,p)

    return dict(
            transition=(S-s),
            arbitrage=res
            )



class Model:
    
    def __init__(self, symbols, calibration_dict, funs, options=None, infos=None):

        self.symbols = symbols

        self.source = dict()
        self.source['functions'] = funs
        self.source['calibration'] = calibration_dict

        
        [calibration, functions] = prepare_model(symbols, calibration_dict, funs)
        self.functions = functions
        self.calibration = calibration
        
        
        self.options = options if options is not None else {}
        self.infos = infos if infos is not None else {}

    def get_calibration(self, pname, *args):

        if isinstance(pname, list):
            return [ self.get_calibration(p) for p in pname ]
        elif isinstance(pname, tuple):
            return tuple( [ self.get_calibration(p) for p in pname ] )
        elif len(args)>0:
            pnames = (pname,) + args
            return self.get_calibration(pnames)

        group = [g for g in self.symbols.keys() if pname in self.symbols[g]]
        try:
            group = group[0]
        except:
            raise Exception('Unknown symbol {}.')
        i = self.symbols[group].index(pname)
        v = self.calibration[group][i]

        return v


    def set_calibration(self, *args, **kwargs):

        # raise exception if unknown symbol ?

        if len(args)==2:
            pname, pvalue = args
            if isinstance(pname, str):
                self.set_calibration(**{pname:pvalue})
        else:
            # else ignore pname and pvalue
            calib =  self.source['calibration']
            calib.update(kwargs)
            self.calibration = read_calibration(self.symbols, calib)
    
    def __str__(self):
        
        s = '''Model object:
- name: "{name}"
- type: "{type}"
- file: "{filename}\n'''.format(**self.infos)
        
        import pprint
        s += '- residuals:\n'
        s += pprint.pformat(compute_residuals(self),indent=2, depth=1)

        return s

        
    


def prepare_model(symbols, calibration_dict, funs):
    
    import inspect

    calibration = read_calibration(symbols, calibration_dict)
    functions = dict()
    
#    for k,f in funs.iteritems():
    for k,f in funs.items():

        argspec = inspect.getargspec(f)
        if k == 'transition':
            size_output = len( symbols['states'] )
            k = 'transition'
        elif k == 'arbitrage':
            size_output = len( symbols['controls'] )
            k = 'arbitrage'
        else:
            continue

        functions[k] = allocating_function( f, size_output )

    return [calibration,functions]
    
def import_model(filename):

#    d = {}
#    e = {}
    d = {}
    e = {}
    with open(filename) as f:
        code = compile(f.read(), filename, "exec")
    
    # TODO improve message
    exec(code, e, e)

    symbols = e['symbols']
    calibration_dict = e['calibration_dict']
    funs = {
            "transition": e["transition"],
            "arbitrage": e["arbitrage"],
            "markov_chain": e["markov_chain"],
        }
    if "complementarities" in e:
        funs[ "complementarities"] = e["complementarities"]

    model_type = e['model_type']
    model_name = e.get('name')
    if model_name is None:
        model_name = 'anonymous'

    infos = dict()
    infos['filename'] = filename
    infos['type'] = model_type
    infos['name'] = model_name

    if 'options' in e:
        options = e['options']
    else:
        options = {}


    model = Model(symbols, calibration_dict, funs, options, infos)

    return model

    
import numpy as np

def mlinspace(smin,smax,orders):

    if len(orders) == 1:
        res = np.atleast_2d( np.linspace(np.array(smin),np.array(smax),np.array(orders)) )
    else:
        meshes = np.meshgrid( *[np.linspace(smin[i],smax[i],orders[i]) for i in range(len(orders))], indexing='ij' )
        res = np.row_stack( [l.flatten() for l in meshes])
    return numpy.ascontiguousarray(res.T)


import numpy
from numpy import array, zeros

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
            eval_UB_spline(self.a, self.b, self.orders, coefs, points, out)
            return out


def eval_UB_spline(a, b, orders, coefs, s, out=None):

    if out == None:
        inplace = False
        out = zeros(coefs.shape[0])
    else:
        inplace = True

    d = len(a)
    from splines_numba import Ad, dAd
    if d == 1:
        from splines_numba import eval_UBspline_1 as evalfun
    elif d == 2:
        from splines_numba import eval_UBspline_2 as evalfun
    elif d == 3:
        from splines_numba import eval_UBspline_3 as evalfun

    evalfun(a, b, orders, coefs, s, out, Ad, dAd)

    if not inplace:
        return out

        

#    def set_values(self, i, vals):
#
#        from dolo.numeric.interpolation.splines_filter import splines_filter
#
#        n_x = vals.shape[1] # number of variables 
#        if self.__values__ is not None:
#            assert(n_x==self.__values__.shape[2])
#        else:
#            dims = [self.n_m] + self.orders.tolist() + [n_x]
#            dims = array(dims, dtype=int)
#            self.__values__ = zeros(dims)
#            self.__fvalues__ = zeros(dims+2) # filtered values
#            self.n_x = n_x
#        assert(self.N == vals.shape[0])
#
#        
#        coefs = splines_filter(self.a, self.b, self.orders, vals)




        




