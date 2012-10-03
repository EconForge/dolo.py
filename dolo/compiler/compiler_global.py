"""
symbolic model -> compiled model

"""


from dolo.compiler.compiler_functions import model_to_fg
from dolo.misc.caching import memoized

import numpy


class CModel:
    """
    something to document
    """

    model_type = 'fg'

    def __init__(self,model,substitute_auxiliary=False, keep_auxiliary=True, solve_systems=True, compiler='numpy'):
        "something to document"

        self.model = model

        self.states = [str(v) for v in model['variables_groups']['states']]
        self.controls = [str(v) for v in model['variables_groups']['controls']]
        self.shocks = [str(v) for v in model['parameters_ordering']]
        self.parameters = [str(v) for v in model['parameters_ordering']]

        self.__compiler__ = compiler

        if not keep_auxiliary:
            [f,g] = model_to_fg(model,substitute_auxiliary=substitute_auxiliary,solve_systems=solve_systems, compiler=compiler)
            self.__f__ = f
            self.__g__ = g
        else:
            [f,g,a] = model_to_fg(model,substitute_auxiliary=substitute_auxiliary,solve_systems=solve_systems, compiler=compiler)
            self.__f__ = f
            self.__g__ = g
            self.__a__ = a
            self.auxiliaries = [str(v) for v in model['variables_groups']['auxiliary']]


    def g(self,s,x,e,p,derivs=False):
        if self.__compiler__ == 'numpy':
            # evertyhing is ready
            return self.__g__(s,x,e,p,derivs=derivs)
        elif derivs is False:
            return self.__g__(s,x,e,p)
        else:
            g_0 = self.__g__(s,x,e,p)
            g_s = numdiff( lambda l: self.__g__(l,x,e,p), s, g_0)
            g_x = numdiff( lambda l: self.__g__(s,l,e,p), x, g_0)
            g_e = numdiff( lambda l: self.__g__(s,x,l,p), e, g_0)
            return [g_0, g_s, g_x, g_e]


    def f(self,s,x,S,X,E,p,derivs=False):
        if self.__compiler__ == 'numpy':
            return self.__f__(s,x,S,X,E,p,derivs=derivs)
        elif derivs is False:
            return self.__f__(s,x,S,X,E,p)
        else:
            f_0 = self.__f__(s,x,S,X,E,p)
            f_s = numdiff(lambda l: self.__f__(l,x,S,X,E,p), s, f_0)
            f_x = numdiff(lambda l: self.__f__(s,l,S,X,E,p), x, f_0)
            f_S = numdiff(lambda l: self.__f__(s,x,l,X,E,p), S, f_0)
            f_X = numdiff(lambda l: self.__f__(s,x,S,l,E,p), X, f_0)
            f_E = numdiff(lambda l: self.__f__(s,x,S,X,l,p), X, f_0)
            return [f_0, f_s, f_x, f_S, f_X, f_E]


    def a(self,s,x,p,derivs=False):
        if self.__compiler__ == 'numpy':
            return self.__a__(s,x,p,derivs=derivs)
        elif derivs is False:
            return self.__a__(s,x,p)
        else:
            a_0 = self.__a__(s,x,p)
            a_s = numdiff(lambda l: self.__f__(l,x,p), s, f_0)
            a_x = numdiff(lambda l: self.__f__(s,l,p), x, f_0)
            return [a_0, a_s, a_x]


    @property
    @memoized
    def x_bounds(self):

        # TODO : bounds should be compiled in __init__ (after sgm cleanup)


        model = self.model
        complementarities_tags = [eq.tags.get('complementarity') for eq in model['equations_groups']['arbitrage']]
        import re
        regex = re.compile('(.*)<=(.*)<=(.*)')
        parsed  = [ [model.eval_string(e) for e in regex.match(s).groups()] for s in complementarities_tags]
        lower_bounds_symbolic = [p[0] for p in parsed]
        controls = [p[1] for p in parsed]
        upper_bounds_symbolic = [p[2] for p in parsed]
        try:
            controls == model['variables_groups']['controls']
        except:
            raise Exception("Order of complementarities does not match order of declaration of controls.")
        states = model['variables_groups']['states']
        parameters = model.parameters
        from dolo.compiler.compiling import compile_multiargument_function
        lb = compile_multiargument_function( lower_bounds_symbolic, [states], ['s'], parameters, fname='lb')
        ub = compile_multiargument_function( upper_bounds_symbolic, [states], ['s'], parameters, fname='ub' )
        return [lb,ub]

    def as_type(self,model_type):
        if model_type == 'fg':
            return self
        else:
            raise Exception('Model of type {0} cannot be cast to model of type {1}'.format(self.model_type, model_type))


def numdiff(f,x0,f0=None):

    eps = 1E-6

    if f0 == None:
        f0 = f(x0)

    p = f0.shape[0]
    q = x0.shape[0]
    N = x0.shape[1]

    df = numpy.zeros( (p,q,N) )
    for i in range(q):
        x = x0.copy()
        x[i,:] += eps
        ff = f(x)
        df[:,i,:] = (ff - f0)/eps

    return df
