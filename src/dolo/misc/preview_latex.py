from dolo.model.symbolic import Variable, Parameter, Shock, Equation

#import os
#import time
#import tempfile
#from latex import latex


from sympy.printing.latex import LatexPrinter as SLP
from sympy.printing.latex import Printer as SP
#from sympy.printing.printer import Printer

class LatexPrinter(SLP):

    #def _print_Equation(self, expr):
    #	return(r"this is an equation")
    def doprint(self, expr):
#        #if isinstance(expr,)'
        print expr
        #tex = SLP().doprint(expr)
        tex = SLP.doprint(self,expr)
        print tex
        #tex = self.doprint( expr)
        return r"%s" % tex
    
    def _print_list(self, expr):
        #txt = "\\begin{eqnarray*}%s\\end{eqnarray*}" % str.join("\\\\",[self._print(i) for i in expr])
        txt = str.join("\n\\\\",["%s" %self._print(i) for i in expr])
        return(txt)
    
    def print_equations(self,expr):
        txt = "\section*{Equations}\n"
        txt = txt +  str.join("\\\\\n", ["%s %s" %(i.name,self._print(i)) for i in expr])
        return(txt)
    
    def _print_Equation(self, expr):
        #return "%s $$%s$$ " %(expr.name, self._print_Relational(expr))
        return( "$$%s$$" %self._print_Relational(expr) )
    

def latex(expr):
    return LatexPrinter().doprint(expr)
