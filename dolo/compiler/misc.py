from collections import OrderedDict
import numpy
from numpy import array, zeros

from interpolation.splines.filter_cubic import filter_data


def filter(smin, smax, orders, controls):

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

    from dolo.compiler.triangular_solver import solve_triangular_system

    sol = solve_triangular_system(calibration_dict)

    calibration  = OrderedDict()
    for group in symbols:
        calibration[group] = numpy.array(
                                [sol[s] for s in symbols[group]],
                             dtype=float)

    return calibration


def calibration_to_dict(symbols, calib):

    from collections import OrderedDict
    if not isinstance(symbols, dict):
        symbols = symbols.symbols

    d = OrderedDict()
    for group, values in calib.items():
        if group == 'covariances':
            continue
        syms = symbols[group]
        for i,s in enumerate(syms):
            d[s] = values[i]

    return d

from collections import OrderedDict
from dolo.compiler.misc import calibration_to_dict

import copy

class CalibrationDict(OrderedDict):

    # cb = CalibrationDict(symbols, calib)
    # calib['states'] -> array([ 1.        ,  9.35497829])
    # calib['states','controls'] - > [array([ 1.        ,  9.35497829]), array([ 0.23387446,  0.33      ])]
    # calib['z'] - > 1.0
    # calib['z','y'] -> [1.0, 0.99505814380953039]

    def __init__(self, symbols, calib):
        superclass = super()
        calib = copy.deepcopy(calib)
        for v in calib.values():
            v.setflags(write=False)
        superclass.__init__(calib)
        self.symbols = symbols
        self.full = calibration_to_dict(symbols, calib)

    def __getitem__(self, p):
        if isinstance(p,tuple):
            return [self[e] for e in p]
        if p in self.symbols.keys():
            return super().__getitem__(p)
        else:
            return self.full[p]


def allocating_function(inplace_function, size_output):

    def new_function(*args, **kwargs):
        val = numpy.zeros(size_output)
        nargs = args + (val,)
        inplace_function( *nargs )
        if 'diff' in kwargs:
            return numdiff(new_function, args)
        return val

    return new_function


def numdiff(fun, args):

    """Vectorized numerical differentiation"""

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



def check(model, silent=False):

    from numpy import concatenate
    from numpy.testing import assert_almost_equal

    from collections import OrderedDict

    checks = OrderedDict()

#    for name in model.calibration.keys():
#        assert( [len(model.calibration[name])==len(model.symbols[name])] )

    names = ['markov_states', 'states', 'controls', 'parameters']

    m,s,x,p = model.calibration['markov_states', 'states', 'controls', 'parameters']


    # check steady_state
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
