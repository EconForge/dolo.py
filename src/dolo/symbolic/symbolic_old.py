import sympy
from sympy import Symbol,Equality,Matrix,latex
from sympy import log,exp,sin,cos,zero
from xml.etree import ElementTree as ET
import copy
#import inspect
#import re

#   Symbol
# (sympy.Symbol)
#      |
#      -------------------------------
#      |                             |
#   SSymbol                      Parameter
# (subscripted symbol)
#      |
#      ---------------
#      |             |
#   Variable       Shock

class Parameter(sympy.Symbol):

    def __init__(self, name, latex_name=None):
        super(Parameter,self).__init__()
        self.name = name
        self.latex_name = latex_name
        return(None)

    def __repr__(self):
        return(self.name)

    def toxml(self):
        return ET.Element('parameter',{'name':self.name})

    def _latex_(self):
        '''Returns latex representation for Sage'''
        return( self.__latex__() )

    def __latex__(self):
        if self.latex_name <> None:
            return(self.latex_name)
        else:
            txt = sympy.latex(sympy.Symbol(self.name)).strip("$")
        return(txt)

class TSymbol(sympy.Symbol):

    def __init__(self, name, latex_name=None):
        super(TSymbol,self).__init__()
        self.basename = name
        self.lag = 0
        self.__latex_name = latex_name
        #self.parent = self # this is not very elegant ! find a better way for singleton classes
        return(None)

    def __call__(self,lag):
        basename = str(self.basename)
        if self.lag == 'S':
            raise Exception,"lags are not allowed on steady state variable"
        #elif lag == -self.lag:
        #    return self.parent
        else:
            newlag = self.lag + lag
            # following is a workaround for sympy calling StrPrint.doPrint method
            if newlag < 0:
                newname = basename + "{-" + str(- newlag) +"}"
            #newname = self.basename + "(" + str( newlag) + ")"
            elif newlag > 0:
                newname = basename + "{" + str( newlag ) + "}"
            #newname = self.basename + "(" + str( newlag ) + ")"
            else:
                newname = basename

            v = self.__class__(newname)
            v.lag = newlag
            v.basename = basename
            #v.parent = self.parent
            #print(v.latex_name)
        return v

    #def getlatex_name(self):
    #    return(self.parent.__latex_name)
    
    #def setlatex_name(self,latex_name):
#        if self.lag != 0:
#            raise Exception('You can change latex names only for symbols without lag')
#        else:
#            self.__latex_name = latex_name
#    latex_name = property(getlatex_name,setlatex_name)

    #def __repr__(self):
    #    self.precedence=-1
    #    return(self.tostr())

    def __str__(self):
        if self.lag == "S":
            result = self.basename + ".S"
        elif self.lag == 0:
            result = self.basename
        else:
            result = self.basename + "(" + str(self.lag) + ")"
        return result

    def __repr__(self):
        return self.__str__()


    def tostr(self, level=0):
        precedence = self.precedence
        result = self.__str__()
        if precedence<=level:
            return('(%s)' % (result))
        return result

#    def toxml(self):
#        if self.lag == 0:
#            return ET.Element('variable',{'name':self.basename})
#        else:
#            return ET.Element('variable',{'name':self.basename,'lag':str(self.lag)})
#
#    def _latex_(self):
#        '''Returns latex representation for Sage'''
#        return( self.__latex__() )
#
#    def __latex__(self):
#        #return("me")
#        latex_base = self.parent.latex_name
#        if latex_base == None:
#            latex_base = self.basename
#        #if self.latex_name <> None:
#        #    latex_base = self.latex_name
#        #else:
#        #    latex_base = self.basename
#        if latex_base.find('@t') == -1:
#            if self.lag == "S":
#                return("\\overline{%s}" %latex_base)
#            elif self.lag == 0:
#                return(latex_base)
#            elif self.lag > 0:
#                return(latex_base + "(+" + str(self.lag) + ")" )
#            elif self.lag < 0:
#                return(latex_base + "(-" + str(-self.lag) + ")" )
#        else:
#            if self.lag == 0:
#                res = latex_base.replace('@t', 't')
#            elif self.lag == 'S':
#                raise Exception("We don't know how to print steady state variable")
#            else:
#                if self.lag >0:
#                    res = latex_base.replace('@t', 't+' + str(self.lag) )
#                else:
#                    res = latex_base.replace('@t', 't' + str(self.lag) )
#            return res

class Variable(TSymbol):
    """
    Define a new variable with : x = Variable('x')
    Then x(k) returns the value with lag k
    Internally x(-k) is a symbol with name x_bk
    """

    def __init__(self, name, latex_name=None):
        super(self.__class__,self).__init__( name, latex_name)
        return None

    @property
    def P(self):
        return(self(-self.lag))

    @property
    def S(self):
        newname = self.basename + "_{S}"
        v = Variable(newname)
        v.lag = "S"
        v.basename = self.basename
        return(v)

    def __init__(self, name, latex_name=None):
        super(self.__class__,self).__init__( name, latex_name)
        return None


class Shock(TSymbol):
    """
    Define a new shock with : x = Shock('x')
    Then x(k) returns the value with lag k
    Internally x(-k) is a symbol with name x_bk
    """

    def __init__(self, name, latex_name=None):
        super(self.__class__,self).__init__( name, latex_name)
        return None


class Equation(sympy.Equality):
    
    def __init__(self, lhs, rhs, name=None,is_endogenous=True):
        super(sympy.Equality,self).__init__()
        #self.is_endogenous=is_endogenous
        # infos are computed in the program
        self.infos = {}
        self.infos['n'] = None
        # tags are usually supplied by the user
        self.tags = {}
        if name != None:
            self.tags['name'] = name
    

    @property
    def name(self):
        return(self.tags.get('name'))

    @property
    def n(self):
        return(self.infos.get('n'))
    
    def copy(self):
        # This function doesn't seem to be called
        eq = Equation(copy.copy(self.lhs),copy.copy(self.rhs),copy.copy(self.name))
        eq.n = copy.copy(self.n)
        eq.info = copy.copy(self.info)
    
    #def subs_dict(self,dict):
    def subs(self,dict):
        eq = Equation(self.lhs.subs(dict),self.rhs.subs(dict),copy.copy(self.name))
        eq.tag(**self.tags)
        return eq
    
    def label(self,sdict):
        self.info.update(sdict)
        return self

    def tag(self,**kwargs):
        self.tags.update(kwargs)
        return self
    

    @property
    def gap(self):
        return( self.lhs - self.rhs)

    @property
    def variables(self):
        l = [v for v in self.atoms() if isinstance(v,Variable)]
        return set(l)

    @property
    def shocks(self):
        l = [v for v in self.atoms() if isinstance(v,Shock)]
        return set(l)

    @property
    def parameters(self):
        l = [p for p in self.atoms() if isinstance(p,Parameter)]
        return l


    def _latex_(self):
        terms = []
        for sexp in [self.lhs,self.rhs]:
            vs = [v for v in sexp.atoms() if isinstance(v,Variable)]
            if len(vs) > 0:
                date_max = max([v.lag for v in vs])
            else:
                date_max = 0
            if date_max > 0:
                terms.append( "\\mathbb{E}\\left[ %s \\right]"% sympy.latex(sexp).strip("$") )
            else:
                terms.append(  latex(sexp).strip("$") )
                
        return "%s = %s" % (terms[0],terms[1])
        #from misc.preview  import DDLatexPrinter
        #return( DDLatexPrinter(inline=False).doprint( ddlatex(self) ) )

    def toxml(self):
        # equations are stored as Dynare strings
        xmleq = ET.Element('equation')
        s = str(self)
        s = s.replace("==","=")
        s = s.replace("**","^")
        #s = s.replace("_b1","(-1)") # this should allow lags more than 1
        #s = s.replace("_f1","(+1)") # this should allow lags more than 1
        xmleq.text = str(self)
        return xmleq