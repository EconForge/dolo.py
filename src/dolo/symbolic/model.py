import sympy
from dolo.symbolic.symbolic import Equation,Variable,Shock,Parameter

class Model(dict):


    def __init__(self,*kargs,**kwargs):
        super(Model,self).__init__(self,*kargs,**kwargs)
        self.check()
        self.check_consistency()
        self.__special_symbols__ = [sympy.exp,sympy.log,sympy.sin,sympy.cos,sympy.tan]
        self.__compiler__ = None

    def check(self):

        defaults = {
            'name': 'anonymous',
            'init_values': {},
            'parameters_values': {},
            'covariances': sympy.Matrix(),
            'variables_ordering': [],
            'parameters_ordering': [],
            'shocks_ordering': []
        }

        for k in defaults:
            if k not in self:
                self[k] = defaults[k]

        if not self.get('equations'):
            raise Exception('No equations specified')

        for n,eq in enumerate(self['equations']):
            if not isinstance(eq,Equation):
                self['equations'][n] = Equation(eq,0)

    @property
    def equations(self):
        return self['equations']

    @property
    def covariances(self):
        return self['covariances'] # should get rid of this

    @property
    def parameters_values(self):
        return self['parameters_values'] # should get rid of this

    @property
    def init_values(self):
        return self['init_values'] # should get rid of this

    @property
    def compiler(self):
        if not(self.__compiler__):
            from dolo.compiler.compiler_python import PythonCompiler
            self.__compiler__ = PythonCompiler(self)
        return self.__compiler__

    def reorder(self, vars, variables_order):
        arg = list(vars)
        res = [v for v in variables_order if v in arg]
        t =  [v for v in arg if v not in variables_order]
        t.sort()
        res.extend( t )
        return res

    def check_consistency(self,verbose=False):

        print_info = verbose
        print_eq_info = verbose

        all_dyn_vars = set([])
        all_dyn_shocks = set([])
        all_parameters = set([])
        for i in range(len(self.equations)):
            eq = self.equations[i]
            eq.infos['n'] = i+1
            atoms = eq.atoms()
            vs = [a for a in atoms if isinstance(a,Variable)]
            ss = [a for a in atoms if isinstance(a,Shock)]
            ps = [a for a in atoms if isinstance(a,Parameter)]
            all_dyn_vars.update(vs)
            all_dyn_shocks.update(ss)
            all_parameters.update(ps)
        tv = [v.P for v in all_dyn_vars]
        ts = [s.P for s in all_dyn_shocks]
        tp = [p for p in all_parameters]
        [tv,ts,tp] = [list(set(ens)) for ens in [tv,ts,tp]]

        self.variables = self.reorder(tv,self['variables_ordering'])
        self.shocks = self.reorder(ts,self['shocks_ordering'])
        self.parameters = self.reorder(tp,self['parameters_ordering'])

        info = {
                "n_variables" : len(self.variables),
                "n_parameters" : len(self.parameters),
                "n_shocks" : len(self.shocks),
                "n_equations" : len(self.equations)
        }
        self.info = info
        if verbose:
            print("Model check : " + self['name'])
            for k in info:
                print("\t"+k+"\t\t"+str(info[k]))

    def eval_string(self,string):
        # rather generic method (should be defined for any model with dictionary updated accordingly
        context = dict()
        for v in self['variables_ordering'] + self['parameters_ordering'] + self['shocks_ordering']:
            context[v.name] = v
        for s in self.__special_symbols__:
            context[str(s)] = s
        return sympy.sympify( eval(string,context) )


    @property
    def fname(self):
        return self['name']

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

        return ord

    @property
    def dr_var_order(self):
        dvo = self.dyn_var_order
        purely_backward_vars = [v for v in self.variables if (v(1) not in dvo) and (v(-1) in dvo)]
        purely_forward_vars = [v for v in self.variables if (v(-1) not in dvo) and (v(1) in dvo)]
        static_vars =  [v for v in self.variables if (v(-1) not in dvo) and (v(1) not in dvo) ]
        mixed_vars = [v for v in self.variables if not v in purely_backward_vars+purely_forward_vars+static_vars ]
        dr_order = static_vars + purely_backward_vars + mixed_vars + purely_forward_vars
        return dr_order

    @property
    def state_variables(self):
        return [v for v in self.variables if v(-1) in self.dyn_var_order ]

    def compute_steadystate_values(self):
        model = self
        from dolo.misc.calculus import solve_triangular_system

        dvars = dict()
        dvars.update(model.parameters_values)
        dvars.update(model.init_values)
        for v in model.variables:
            if v not in dvars:
                dvars[v] = 0
        undeclared_parameters = []
        for p in model.parameters:
            if p not in dvars:
                undeclared_parameters.append(p)
                dvars[p] = 0
                raise Warning('No initial value for parameters : ' + str.join(', ', [p.name for p in undeclared_parameters]) )

        values = solve_triangular_system(dvars)[0]

        y = [values[v] for v in model.variables]
        x = [0 for s in model.shocks]
        params = [values[v] for v in model.parameters]
        return [y,x,params]

    def subs(self,a,b):

        if isinstance(a,str):
            a = sympy.Symbol(a)

        nmodel = Model(**self)
        nmodel['equations'] = [eq.subs({a:b}) for eq in nmodel['equations']]
        for k,v in nmodel['init_values'].iteritems():
            if isinstance(v,sympy.Basic):
                nmodel['init_values'][k] = v.subs({a:b})

        nmodel.check()
        return nmodel


def compute_residuals(model):
    [y,x,parms] = model.compute_steadystate_values()
    dd_v = dict([(model.variables[i],y[i]) for i in range(len(y))])
    dd_p = dict([(model.parameters[i],parms[i]) for i in range(len(parms))])
    dd_x = dict([(v,0) for v in model.shocks])
    dd = dict()
    dd.update(dd_v)
    dd.update(dd_p)
    dd.update(dd_x)
    stateq = [ eq.subs( dict([[v,v.P] for v in eq.variables]) ) for eq in model.equations]
    stateq = [ eq.subs( dict([[v,0] for v in eq.shocks]) ) for eq in stateq]
    stateq = [ eq.rhs - eq.lhs for eq in stateq ]
    residuals = [ float(eq.subs(dd)) for eq in stateq ]
    return residuals

if __name__ == '__main__':

    from dolo.symbolic.symbolic import Variable,Equation

    v = Variable('v',0)

    eq = Equation( v**2, v(1) - v(-1))

    d = Model(equations=[eq])
