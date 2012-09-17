from dolo.compiler.compiler_functions import model_to_fga, model_to_fg

from dolo.numeric.serial_operations import serial_multiplication as smult

from dolo.misc.caching import memoized

class CModel_fg:

    model_type = 'fg'

    def __init__(self,model,substitute_auxiliary=False, solve_systems=False):

        self.model = model

        [f,g] = model_to_fg(model,substitute_auxiliary=substitute_auxiliary,solve_systems=solve_systems)
        self.f = f
        self.g = g



#    def g(self,s,x,e,p, derivs=False):
#        return self.__g__(s,x,e,p,derivs=derivs)
#
#    def f(self,s,x,S,X,e,p,derivs=False):
#        return self.__f__(s,x,S,X,e,p,derivs=derivs)

    @property
    @memoized
    def x_bounds(self):
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
        from dolo.compiler.compiling import compile_function_2
        lb = compile_function_2( lower_bounds_symbolic, [states], ['s'], parameters, fname='lb')
        ub = compile_function_2( upper_bounds_symbolic, [states], ['s'], parameters, fname='ub' )
        return [lb,ub]

    def as_type(self,model_type):
        if model_type == 'fg':
            return self
        else:
            raise Exception('Model of type {0} cannot be cast to model of type {1}'.format(self.model_type, model_type))


class CModel_fga:

    model_type = 'fga'

    def __init__(self,model):
        self.model = model

        [f,a,g] = model_to_fga(model)
        self.__f = f
        self.__a = a
        self.__g = g

    def as_type(self,model_type):
        if model_type == 'fg':
            return self
        elif model_type == 'fga':
            return self
        else:
            raise Exception('Model of type {0} cannot be cast to model of type {1}'.format(self.model_type, model_type))
        return

    def g(self,s,x,e,p,derivs=False):
        if not derivs:
            a = self.__a(s,x,p,derivs=False)
            return self.__g(s,x,a,e,p,derivs=False)
        else:
            [a,a_s,a_x] = self.__a(s,x,p,derivs=True)
            [g,g_s,g_x,g_a,g_e] = self.__g(s,x,a,e,p,derivs=True)
            G = g
            G_s = g_s + smult(g_a,a_s)
            G_x = g_x + smult(g_a,a_x)
            G_e = g_e
            return [G,G_s,G_x,G_e]

    def a(self,s,x,p,derivs=False):
        return self.__a(s,x,p,derivs=derivs)


    def f(self, s, x, snext, xnext, e, p, derivs=False):


        if not derivs:
            a = self.__a(s,x,p,derivs=False)
            anext = self.__a(snext,xnext,p,derivs=False)
            return self.__f(s,x,snext,xnext,a,anext,e,p,derivs=False)
        else:
            [a,a_s,a_x] = self.__a(s,x,p,derivs=True)
            [A,A_S,A_X] = self.__a(snext,xnext,p,derivs=True)
            [f,f_s,f_x,f_S,f_X,f_a,f_A] = self.__f(s,x,snext,xnext,a,A,e,p)
            F = f
            F_s = f_s + smult(f_a,a_s)
            F_x = f_x + smult(f_a,a_x)
            F_S = f_S + smult(f_A,A_S)
            F_X = f_X + smult(f_A,A_X)
            return [F,F_s,F_x,F_S,F_X]


# for compatibility

CModel = CModel_fg
CModel2 = CModel_fga

GlobalCompiler = CModel
GlobalCompiler2 = CModel2

#from global_solution import *