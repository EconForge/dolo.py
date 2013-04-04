"""
Compiled models.

"""

import sympy


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

def solve_recursive_block(equations):
    from dolo.symbolic.symbolic import Equation
    system = {eq.lhs: eq.rhs for eq in equations}
    from dolo.misc.triangular_solver import solve_triangular_system
    sol = solve_triangular_system(system)
    solved_equations = [Equation(eq.lhs, sol[eq.lhs]) for eq in equations]
    return solved_equations