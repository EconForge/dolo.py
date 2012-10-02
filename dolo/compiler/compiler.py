"""
Compiled models.

Several kinds of models


"""


from dolo.symbolic.model import *
from dolo.symbolic.symbolic import *

import numpy as np

class Compiler:
    model = None
    
    def __init__(self,model):
        self.model = model
        self._dynamic_substitution_list = None
        self._static_substitution_list = None
        self.covariances = self.model.covariances

    def dynamic_substitution_list(self,brackets=False,compact=True):
        # Returns a dictionary mapping each symbol appearing in the model
        # to the string used in the dynamic file

        if compact == False:
            dyn_order = [v(1) for v in self.model.variables]
            dyn_order += self.model.variables
            dyn_order += [v(-1) for v in self.model.variables]
        else:
            dyn_order = self.model.dyn_var_order

        if brackets:
            ind_0 = 0
        else:
            ind_0 = 1
        if not brackets:
            subs_dict = dict()
            for i,v in enumerate(dyn_order):
                print v
                #if v.P in self.model.variables:
                subs_dict[v] = 'y({0})'.format(i + 1)
            for i in range(len(self.model.shocks)):
                s = self.model.shocks[i]
                subs_dict[s] = 'x(it_,{0})'.format(i + 1)
            for i in range(len(self.model.parameters)):
                p = self.model.parameters[i]
                subs_dict[p] = 'params({0})'.format(i + 1)
            return subs_dict
        else:
            subs_dict = dict()

            for i,v in enumerate( dyn_order):
                #if v.P in self.model.variables:
                subs_dict[v] = 'y[{0}]'.format(i + ind_0)
            for i in range(len(self.model.shocks)):
                s = self.model.shocks[i]
                subs_dict[s] = 'x[it_,{0}]'.format(i + ind_0)
            for i in range(len(self.model.parameters)):
                p = self.model.parameters[i]
                subs_dict[p] = 'params[{0}]'.format(i + ind_0)
            return subs_dict

    #@property
    def static_substitution_list(self,y='y',x='x',params='params',ind_0=1,brackets=False): # what is the meaning of these options ?
        # Returns a dictionary mapping each symbol appearing in the model
        # to the string used in the static file
        if self._static_substitution_list != None:
            return self._static_substitution_list
        else:
            if not brackets:
                subs_dict = dict()
                var_order = self.model.variables
                for i in range(len(var_order)):
                    v = var_order[i]
                    subs_dict[v] = '{y}({v})'.format(y=y,v=i + ind_0)
                for i in range(len(self.model.shocks)):
                    s = self.model.shocks[i]
                    subs_dict[s] = '{x}({v})'.format(x=x,v=i + ind_0)
                for i in range(len(self.model.parameters)):
                    p = self.model.parameters[i]
                    subs_dict[p] = '{params}({v})'.format(params=params,v=i + ind_0)
                return subs_dict
            else:
                subs_dict = dict()
                var_order = self.model.variables
                for i in range(len(var_order)):
                    v = var_order[i]
                    subs_dict[v] = '{y}[{v}]'.format(y=y,v=i + ind_0)
                for i in range(len(self.model.shocks)):
                    s = self.model.shocks[i]
                    subs_dict[s] = '{x}[{v}]'.format(x=x,v=i + ind_0)
                for i in range(len(self.model.parameters)):
                    p = self.model.parameters[i]
                    subs_dict[p] = '{params}[{v}]'.format(params=params,v=i + ind_0)
                return subs_dict

            
    def lead_lag_incidence_matrix(self):
        columns = dict()
        model = self.model
        for eq in self.model.equations:
            all_vars = [a for a in eq.atoms() if isinstance(a,Variable)]
            for e in all_vars:
                if not e.lag in columns.keys():
                    columns[e.lag] = set()
                columns[e.lag].add(e)
        n_endos = len(self.model.variables)
        minimum_endo_lag = abs(min(columns.keys()))
        maximum_endo_lag = abs(max(columns.keys()))
        lagged_variables = []
        for l in sorted( columns.keys() ):
            x = [model.variables[k](l) for k in range(len(model.variables)) if model.variables[k](l) in columns[l]]
            lagged_variables.extend(x)
            #lagged_variables.extend(  [model.variables[k] for k in range(len(model.variables)) if model.variables[k](l) in columns[l]]   )
        mat = np.zeros((n_endos , 1 + minimum_endo_lag + maximum_endo_lag),dtype=np.int8)
        for v in lagged_variables:
            i = model.variables.index(v.P)
            j = v.lag + minimum_endo_lag
            mat[i,j] = int(lagged_variables.index(v) + 1)
        return mat.T

    def static_tabify_expression(self,eq,for_matlab=False,for_c=True):
        #from dolo.misc.calculus import map_function_to_expression
        # works when all variables are expressed without lag
        subs_dict = self.static_substitution_list
        res = DicPrinter(subs_dict).doprint(eq)

    def dyn_tabify_expression(self,eq,for_matlab=False,for_c=True):
        #from dolo.misc.calculus import map_function_to_expression
        # works when all variables are expressed without lag
        subs_dict = self.dynamic_substitution_list
        res = DicPrinter(subs_dict).doprint(eq)
#        def f(expr):
#            if expr.__class__ in [Variable,Parameter]:
#                    vname = subs_dict[expr]
#                    return(Symbol(vname))
#            else:
#                return(expr)
#        res = map_function_to_expression(f,eq)
        return res

    def tabify_expression(self,eq,for_matlab=False,for_c=True):
        #from dolo.misc.calculus import map_function_to_expression
        # works when all variables are expressed without lag
        subs_dict = self.build_substitution_list(for_matlab,not for_c)
        def f(expr):
            if expr.__class__ in [Variable,Parameter]:
                    vname = subs_dict[expr]
                    return(Symbol(vname))
            else:
                return(expr)
        res = map_function_to_expression(f,eq)
        return res
    
    def get_gaps_static_python(self):
        import scipy
        current_equations = []
        current_equations = self.model.dss_equations()
        tab_eq = []
        for eq in current_equations:
            #if eq.is_endogenous:
            tab_eq.append(self.tabify_expression(eq))
        # count endogenous equations (should be in Model)s:
        code  = "def gaps_static(x,y,parm):\n"
        #code += "    print('x',x)\n"
        #code += "    print('y',y)\n"
        #code += "    print('parm',parm)\n"
        code += "    gaps = scipy.zeros(%s)\n" % len(tab_eq)
        i = 0
        for eq in tab_eq:
            eq_s = str(eq.gap())
            code += "    gaps[" + str(i) + "] = " + eq_s + "\n"
            i = i + 1
        #code += "    print('gaps :', gaps)\n"
        code += "    return gaps"
        
        # function has been modified so as to return source code
        return(code)
        #print(code)
        #exec(code)
        #import psyco
        #psyco.bind(gaps_static)    
        #return(gaps_static)
    
    def write_gaps_ccode_static(self):
        current_equations = self.model.current_equations()
        tab_eq = map(self.tabify_expression, current_equations)
        code = "py::tuple gaps(" + str(len(tab_eq)) + ");\n"
        for i in range(len(tab_eq)):
            eq_s = str(tab_eq[i].gap()).replace('**','^') # to be fixed with function pow()
            code += "gaps[" + str(i) + "] = " + eq_s + ";\n"
        code += "return_val = gaps;"
        
        return(code)


class DicPrinter(sympy.printing.StrPrinter):

    def __init__(self,printing_dict):
        super(DicPrinter,self).__init__()
        self.printing_dict = printing_dict

    def doprint_matlab(self,expr,vectorize=False):
        txt = self.doprint(expr)
        txt = txt.replace('**','^')
        if vectorize:
            txt = txt.replace('^','.^')
            txt = txt.replace('*','.*')
            txt = txt.replace('/','./')
            #txt = txt.replace('+','.+')
            #txt = txt.replace('-','.-')

        return txt

    def doprint_numpy(self,expr,vectorize=False):
        txt = self.doprint(expr)
        return txt


    def _print_Symbol(self, expr):
        return self.printing_dict[expr]

    def _print_TSymbol(self, expr):
        return self.printing_dict[expr]

    def _print_Variable(self, expr):
        return self.printing_dict[expr]

    def _print_Parameter(self, expr):
        return self.printing_dict[expr]

    def _print_Shock(self, expr):
        if expr in self.printing_dict:
            return self.printing_dict[expr]
        else:
            return str(expr)
