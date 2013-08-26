
import sympy
from dolo.symbolic.symbolic import Equation,Variable,Shock,Parameter
from collections import OrderedDict

import numpy  # missing: option to disable numpy completely
floatX = numpy.float64
nan = numpy.nan

class SModel:

    fname = 'anonymous'
    name = 'anonymous'

    __data__ = None # may contain the data used to initialize the model
    __special_symbols__ = [sympy.exp,sympy.log,sympy.sin,sympy.cos,sympy.tan, sympy.asin, sympy.acos, sympy.atan, sympy.sqrt,sympy.Symbol('inf')]

    def __init__(self, equations_groups, symbols_s, calibration_s=None, covariances_s=None):

        if isinstance(equations_groups, list):
            equations_groups = {'single_block': equations_groups}

        if calibration_s is None:
            calibration_s = {}

        if covariances_s is None:
            n = len(symbols_s['shocks'])
            covariances_s = sympy.zeros( (n,n) )

        initialized_to_zero = ['shocks','variables']
        for vt in initialized_to_zero:
            l = symbols_s.get(vt)
            if l is None:
                continue
            for s in  l: # TODO : issue a warning for non initialized values
                if s not in calibration_s:
                    calibration_s[s] = 0

        #######################################
        # the model object is defined by:
        self.equations_groups = OrderedDict( (k,v) for k,v in equations_groups.iteritems())     # dict: string -> (list: Equation)
        self.symbols_s = OrderedDict( (k,v) for k,v in symbols_s.iteritems() )                  # dict: string -> (list: sympy)
        self.calibration_s = calibration_s                                                      # dict: sympy -> sympy
        self.covariances_s = covariances_s                                                      # sympy matrix
        ######################################


        # this should actually be non deterministic
        import random
        n = random.random()
        self.__hashno__ = hash(n)

        self.update()

    def check():
        '''Tests whether symbolic model is well defined'''
        pass


    def update(self):

        '''Propagates changes in symbolic structure'''
        self.symbols = OrderedDict( (k, tuple(str(h) for h in v) ) for k,v in self.symbols_s.iteritems() )                  # dict: string -> (list: sympy)

        l = []
        for e in self.symbols_s.keys():
            if e not in ('shocks','parameters'):
                l.extend(self.symbols_s[e])

        # for backward compatibility
        self.variables = l
        self.shocks = self.symbols_s['shocks']
        self.parameters = self.symbols_s['parameters']


        # update calibration
        from dolo.misc.triangular_solver import solve_triangular_system
        calibration_dict = solve_triangular_system(self.calibration_s) # calibration dict : sympy -> float
        self.calibration_dict = calibration_dict

        calibration = OrderedDict()  # calibration dict (by group) : string -> ( sympy -> float )

        for vg in self.symbols_s:
            vars = self.symbols_s[vg]
            values = [ (float(calibration_dict[v]) if v in calibration_dict else nan) for v in vars]
            calibration[vg] = numpy.array( values, dtype=floatX )

        sigma = self.covariances_s.subs(self.calibration_s)
        sigma = numpy.array( sigma ).astype( floatX )

        calibration['covariances'] = sigma
        self.calibration = calibration
        self.sigma = sigma


        l = []
        for eqg,eq in self.equations_groups.iteritems():
            l.extend(eq)
        self.equations = l

    def set_calibration(self,d):
        dd = {}
        for k in d:
            if isinstance(k,str):
                if k in self.symbols['parameters']:
                    kk = Parameter(k)
                else:
                    kk = Variable(k)
            else:
                kk = k
            dd[kk] = d[k]
        self.calibration_s.update(dd)
        self.update()

    def __hash__(self):
        return self.__hashno__


    def eval_string(self,string):
        # rather generic method (should be defined for any model with dictionary updated accordingly
        context = dict()
        for v in self.variables + self.parameters + self.shocks:
            context[v.name] = v
        for s in self.__special_symbols__:
            context[str(s)] = s
        return sympy.sympify( eval(string,context) )

    def copy(self):

        from copy import copy,  deepcopy
        eq_groups = OrderedDict()
        for k in self.equations_groups:
            eg = self.equations_groups[k]
            egg = [eq.copy() for eq in eg]
            eq_groups[k] = egg

        symbols_s = deepcopy(self.symbols_s)
        calibration_s = deepcopy(self.calibration_s)
        covariances_s = deepcopy(self.covariances_s)

        return SModel(eq_groups, symbols_s, calibration_s, covariances_s)


    def __repr__(self):

        res = compute_residuals(self)

        txt = "\nSymbolic model\n"
        txt += "--------------\n\n"

        txt += "Equation blocks:\n"
        for eqg in self.equations_groups:
            txt += '\n\t{}:\n\n'.format(eqg)
            for i,eq in enumerate(self.equations_groups[eqg]):
                if res is not None:
                    r = res[eqg][i]
                    txt += "\t{:=10.5f}\t:\t{}\n".format(r,eq)
                else:
                    txt += "\t\t{}\n".format(eq)
        return txt



    @property
    def dyn_var_order(self):
        # returns a list of dynamic variables ordered as in Dynare's dynamic function
        d = dict()
        for eq in self.equations:
            all_vars = eq.variables
            for v in all_vars:
                if not v.lag in d:
                    d[v.lag] = set()
                d[v.lag].add(v)
        maximum = max(d.keys())
        minimum = min(d.keys())
        ord = []
        for i in range(minimum,maximum+1):
            if i in d.keys():
                ord += [v(i) for v in self.variables if v(i) in d[i]]
        self.__dyn_var_order__ = ord
        return ord


    @property
    def predetermined_variables(self):
        return [v for v in self.variables if v(-1) in self.dyn_var_order ]

    def get_complementarities(self):

        # TODO: currently this works for "arbitrage" equations associated to "controls"

        import re
        regex = re.compile('(.*)<=(.*)<=(.*)')


        model = self

        complementarities_tags = [eq.tags.get('complementarity') for eq in model.equations_groups['arbitrage']]

        parsed  = [ [model.eval_string(e) for e in regex.match(s).groups()] for s in complementarities_tags]
        lower_bounds_symbolic = [p[0] for p in parsed]
        controls = [p[1] for p in parsed]
        upper_bounds_symbolic = [p[2] for p in parsed]
        try:
            controls == model.symbols_s['controls']
        except:
            raise Exception("Order of complementarities does not match order of declaration of controls.")

        complementarities = dict()
        complementarities['arbitrage'] = [lower_bounds_symbolic, upper_bounds_symbolic]

        return complementarities



def iteritems(d):
    return zip(d.keys(), d.values())

def compute_residuals(model):

    dd = model.calibration_dict.copy()
    dd.update( {v(-1): dd[v] for v in model.variables } )
    dd.update( {v(1): dd[v] for v in model.variables } )
    dd.update( {s: 0 for s in model.shocks} )
    dd.update( {v(1): dd[v] for v in model.shocks} )
    dd.update( {v(-1): dd[v] for v in model.shocks} )

    from collections import OrderedDict as odict
    residuals = odict()
    for gname,geqs in iteritems(model.equations_groups):
        l = []
        for eq in geqs:
            if isinstance(eq,Equation):
                t = eq.gap.subs(dd)
            else:
                t = eq
            try:
                t = float(t)
            except Exception as e:
                print('Failed computation of residuals in :\n'+str(eq))
                print('Impossible to evaluate : \n'+str(t))
                raise e
        residuals[ gname ] = [ float( eq.gap.subs( dd ) ) for eq in geqs]
    return residuals
    # else:
    #     stateq = [ eq.gap.subs( dd ) for eq in model.equations]
    #     residuals = [ float(eq) for eq in stateq ]
    #     return residuals




if __name__ == '__main__':

    a = Variable('a')
    b = Variable('b')
    p = Parameter('p')
    s = Shock('s')

    equations = [a + p + b + s]
    calib = {a: 1, p: 0, s: 3.4}
    s_symbols = {'variables': [a, b], 'shocks': [s], 'parameters': [p]}
    # s_calibration = s

    # model = SModel(equations, s_symbols, calib )
    #
    # print(model.calibration)
    #
    #
    # model2 = model.copy()
    #
    # print( model2 == model )
    # print(model.symbols)
    # print(model)

    # filename = '../../examples/dynare_modfiles/example1.mod'
    # filename = '/home/pablo/Documents/Research/CGR/revival/CKM.mod'

    filename = '../../examples/global_models/rbc.yaml'

    # from dolo.misc.modfile import dynare_import
    # model = dynare_import(filename)

    from dolo.misc.yamlfile import yaml_import
    model = yaml_import(filename)


    from dolo import global_solve
    dr = global_solve(model)

    from dolo.numeric.perturbations import solve_decision_rule

    print(model)

    model2 = model.copy()

    model2.set_calibration({'beta':0.95})

    print(model.calibration)
    print(model2.calibration)

    print(model2)
    print(model)

