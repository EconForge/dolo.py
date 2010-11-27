from dolo.symbolic.model import Model
from dolo.symbolic.symbolic import Variable,Equation

import sympy

class RamseyModel(Model):

    def __init__(self, base_model, objective=None, discount=None):
        self.base_model = base_model
        self.variables_ordering = list(base_model.variables_ordering)
        self.parameters_ordering = list(base_model.parameters_ordering)
        self.shocks_ordering = list(base_model.shocks_ordering)
        self.fname = base_model.fname + '_ramsey'
        self.equations = list(base_model.equations)
        self.covariances = base_model.covariances # should be copied instead
        self.init_values = dict(base_model.init_values)
        self.parameters_values = dict(base_model.parameters_values)


        if objective==None or discount==None:
            # parameters must be specified in the model
            ins_eq = [eq for eq in base_model.equations if eq.tags.has_key("ramsey_instrument")]
            if len(ins_eq)==0:
                raise Exception('Some parameters for the policy must be specified in the modfile or the function call.')
            if len(ins_eq)>1:
                raise Exception('Ramsey policy not implemented for many objectives.')
            else:
                ins_eq = ins_eq[0]
            self.instrument_equation = ins_eq
        if ins_eq.tags.has_key('ramsey_objective'):
            objective = ins_eq.tags['ramsey_objective']
        if ins_eq.tags.has_key('ramsey_discount'):
            discount = ins_eq.tags['ramsey_discount']
        if ins_eq.tags.has_key('ignore_variables'):
            ivs = ins_eq.tags['ignore_variables']
            ivs = ivs.replace(',',' ')
            ivs = ivs.squeeze()
            ivs = '[' + ivs.replace(' ',',') + ']'
            self.ignored_variables = self.eval_string(ivs)
            print 'Ignoring : ' + str(self.ignored_variables)
        else:
            self.ignored_variables = []



        self.objective = objective if not isinstance(objective,str) else self.eval_string(objective)
        self.discount = discount if not isinstance(objective,str) else self.eval_string(discount)


        self.build_lagrangian()
        self.fname = self.base_model.fname + '_ramsey'


    def build_lagrangian(self):
        vars = list(self.base_model.variables)

        if self.instrument_equation!=None and self.instrument_equation.tags.has_key('ramsey_instrument'):
            ramsey_instrument = [v for v in vars if v.name == self.instrument_equation.tags['ramsey_instrument']]
            ramsey_instrument = ramsey_instrument[0]
            self.equations.remove(self.instrument_equation)
            vars.remove(ramsey_instrument)
            
        # first we create one multiplicator by equation
        n_eq = len(self.equations)
        lambdas = []
        for i in range(n_eq):
            v = Variable( 'Lambda_' + str(i+1) , 0)
            self.variables_ordering.append(v)
            lambdas.append(v)
        t_cur = self.objective + sum( [self.equations[i].gap * lambdas[i] for i in range(n_eq)] )
        t_fut = time_shift(t_cur,+1)
        t_past = time_shift(t_cur,-1)
        beta = self.discount
        lagrangian = 1/beta * t_past + t_cur + beta * t_fut
        self.lagrangian = lagrangian

        
        print 'There are ' + str(len(self.equations)) + 'equations'
        print 'There are ' + str(len(vars)) + 'variables'

        if self.instrument_equation!=None and self.instrument_equation.tags.has_key('ramsey_instrument'):
#            vars = [ramsey_instrument] + vars
            eq = lagrangian.diff(ramsey_instrument)
            eq = Equation(eq,0).tag( name='Derivative of lagrangian w.r.t : ' + str(ramsey_instrument) )
            eq.tags.update( self.instrument_equation.tags )
            self.equations.append(eq)

        for v in vars:
            if v in self.ignored_variables:
                pass
            eq = lagrangian.diff(v)
            eq = Equation(eq,0).tag(name='Derivative of lagrangian w.r.t : ' + str(v) )
            self.equations.append(eq)

def time_shift(expr,n):
    from dolo.misc.misc import map_function_to_expression
    from dolo.symbolic.symbolic import Variable,Shock
    def f(e):
        if isinstance(e,(Variable,Shock)):
            return e(n)
        else:
            return e
    return map_function_to_expression(f,expr)
