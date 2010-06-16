# To change this template, choose Tools | Templates
# and open the template in the editor.


from dolo.symbolic.model import Model
from dolo.symbolic.symbolic import Variable,Equation
import sympy

class RamseyModel(Model):

    def __init__(self, base_model, objective, discount):
        self.base_model = base_model
        self.variables_ordering = list(base_model.variables_ordering)
        self.parameters_ordering = list(base_model.parameters_ordering)
        self.shocks_ordering = list(base_model.shocks_ordering)
        self.fname = base_model.fname + '_ramsey'
        self.equations = list(base_model.equations)
        self.covariances = base_model.covariances # should be copied instead
        self.init_values = dict(base_model.init_values)
        self.parameters_values = dict(base_model.parameters_values)
        self.objective = objective if not isinstance(objective,str) else self.eval_string(objective)
        self.discount = discount if not isinstance(objective,str) else self.eval_string(discount)
        self.build_lagrangian()

    def eval_string(self,string):
        # rather generic method (should be defined for any model with dictionary updated accordingly
        context = dict()
        for v in self.variables_ordering + self.parameters_ordering + self.shocks_ordering:
            context[v.name] = v
        return sympy.sympify( eval(string,context) )

    def build_lagrangian(self):
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
        for v in self.base_model.variables:
            eq = lagrangian.diff(v)
            eq = Equation(eq,0).tag(name='Derivative of lagrangian w.r.t : ' + str(v) )
            self.equations.append(eq)

def time_shift(expr,n):
    from dolo.misc.misc import map_function_to_expression
    from dolo.model.symbolic import Variable,Shock
    def f(e):
        if isinstance(e,(Variable,Shock)):
            return e(n)
        else:
            return e
    return map_function_to_expression(f,expr)