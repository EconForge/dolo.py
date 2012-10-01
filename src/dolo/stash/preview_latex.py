from sympy.printing.latex import LatexPrinter as SLP

class LatexPrinter(SLP):

    def doprint(self, expr):
        tex = SLP.doprint(self,expr)
        return r"%s" % tex
    
    def _print_list(self, expr):
        #txt = "\\begin{eqnarray*}%s\\end{eqnarray*}" % str.join("\\\\",[self._print(i) for i in expr])
        txt = str.join("\n\\\\",["%s" %self._print(i) for i in expr])
        return(txt)
    
    def print_equations(self,expr):
        txt = "\section*{Equations}\n"
        txt += str.join("\\\\\n", ["%s %s" %(i.name,self._print(i)) for i in expr])
        return(txt)
    
    def _print_Equation(self, expr):
        return( "$$%s$$" %self._print_Relational(expr) )
    

def latex(expr):
    return LatexPrinter().doprint(expr)
