
import sympy

from xml.etree import ElementTree as ET
import copy
import re
import copy


#   Symbol
# (sympy.Symbol)
#      |
#      -------------------------------
#      |                             |
#   TSymbol                      Parameter
# (Time symbol)
#      |
#      ---------------
#      |             |
#   Variable       Shock



# monkey patch sympy so that time symbols are printed correctly

from sympy.printing.str import  StrPrinter
StrPrinter._print_TSymbol = lambda self,x: x.__str__()
from sympy.printing.latex import LatexPrinter
LatexPrinter._print_TSymbol = lambda self,x: x.__latex__()


class Parameter(sympy.Symbol):

    def __repr__(self):
        return(self.name)

    def toxml(self):
        return ET.Element('parameter',{'name':self.name})

    def __call__(self,n):
        print('Parameter {0} has been called. Makes no sense !').format(self)
        return self

    def _latex_(self,*args):
        '''Returns latex representation for Sage'''
        return( self.__latex__() )

    def __latex__(self):
        [rad, indices, exponents] = split_name_into_parts(self.name)
        rad = greekify(rad)
        indices = [greekify(r) for r in indices]
        exponents = [greekify(r) for r in exponents]
        up =  '{' + str.join(',', exponents) + '}'
        down =  '{' + str.join(',', indices) + '}'
        sup = '^{0}'.format(up) if up != '{}'  else ''
        sdown = '^{0}'.format(down) if down != '{}' else ''
        return '{0}{1}{2}'.format(rad,sdown,sup)

class TSymbol(sympy.Symbol):

    #latex_names = {}
    def __init__(self, name, **args):
        super(TSymbol,self).__init__()
        if 'date' not in args:
            self._assumptions['date'] = 0
            self.assumptions0['date'] = 0
        else:
            self._assumptions['date'] = args['date']
            self.assumptions0['date'] = args['date']

        return(None)

    def __call__(self, shift):
        current_date = self.assumptions0['date']
        # we forget other assumptions
        v = type(self)
        return v( self.name, date = current_date + shift)

    def __copy__(self):
        from copy import copy
        v = type(self)
        name = copy(self.name)
        return v( name, date = self.date)


    @property
    def date(self):
        return self.assumptions0['date']

    def _hashable_content(self):
        return (self.name,self.date)

    @property
    def lag(self):
        return self.date

    @property
    def P(self):
        return(self(-self.lag))

    @property
    def S(self):
        v = Variable(self.name,date='S')
#
    def __repr__(self):
        return self.__str__(self)

    def __str__(self):
        if self.lag == "S":
            result = self.name + ".S"
        elif self.lag == 0:
            result = self.name
        else:
            result = self.name + "(" + str(self.lag) + ")"
        return result

    def __repr__(self):
        return self.__str__()


    def tostr(self, level=0):
        precedence = self.precedence
        result = self.__str__()
        if precedence<=level:
            return('(%s)' % (result))
        return result

    def _latex_(self,*args):
        return self.__latex__()

    def __latex__(self):
        '''Returns latex representation for Sage'''
        [rad, indices, exponents] = split_name_into_parts(self.name)
        rad = greekify(rad)
        indices = [greekify(r) for r in indices]
        exponents = [greekify(r) for r in exponents]


        up =  '{' + str.join(',', exponents) + '}'

        if self.date == 'S':
            down = '{' + str.join(',', indices) + '}'
            resp = '{0}_{1}^{2}'.format(rad,down,up)
            resp = '\\overline{' + resp + '}'
            return resp

        elif self.date == 0:
            times = 't'
        elif self.date >0:
            times = 't+' + str(self.date)
        elif self.date <0:
            times = 't-' + str(-self.date)
        else:
            raise(Exception('Time variable {0} has unknown date : {1}.'.format(self.name,self.date)))

        if indices == []:
            down = '{' + times + '}'
        else:
            down =  '{' + str.join(',', indices) + ',' + times + '}'

        if len(up)>2:
            resp = '{0}_{1}^{2}'.format(rad,down,up)
        else:
            resp = '{0}_{1}'.format(rad,down)

        return resp

    def safe_name(self):
        date_string = str(self.date)
        date_string = date_string.replace('-','_')

        return '_{}_{}'.format(date_string, self.name)


# warning: at some point in the future, we will get rid of the distinction between Variable and Shock

class Variable(TSymbol):
    """
    Define a new variable with : x = Variable('x')
    Then x(k) returns the value with lag k
    Internally x(-k) is a symbol with name x_bk
    """
    pass

class Shock(TSymbol):
    """
    Define a new shock with : x = Shock('x')
    Then x(k) returns the value with lag k
    """
    pass

class Equation(sympy.Equality):
    
    def __init__(self, lhs, rhs, name=None):
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


    # def __copy__(self):
    #     return self.__deepcopy__()

    def copy(self):
        # This function doesn't seem to be called

        eq = Equation(copy.copy(self.lhs),copy.copy(self.rhs),copy.copy(self.name))
        # eq.n = copy.copy(self.n)
        # eq.info = copy.copy(self.info)
        ntags = copy.deepcopy(self.tags)
        eq.tag(**ntags)
        return eq

    
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


    def _latex_(self,*args):
        terms = []
        for sexp in [self.lhs,self.rhs]:
            vs = [v for v in sexp.atoms() if isinstance(v,Variable)]
            if len(vs) > 0:
                dates = [v.lag for v in vs]
                if 'S' in dates:
                    dates.remove('S')
                date_max = max(dates)
            else:
                date_max = 0
            if date_max > 0:
                terms.append( "\\mathbb{E}_t\\left[ %s \\right]"% sympy.latex(sexp).strip("$") )
            else:
                terms.append(  sympy.latex(sexp).strip("$") )
                
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




# TODO : expereiments with indexed symbols
# subscripts are denoted by @, upperscripts by $

class IndexedSymbol(sympy.Basic):

    def __init__(self, name, symbol_type):
        super(IndexedSymbol,self).__init__()
        self.name = name
        self.symbol_type = symbol_type


        self.nargs = name.count('@')
        self.basename = self.name.strip('@')
        if self.nargs == 0:
            raise Exception("Indexed symbols must have at least one index.")
        return(None)

    def __getitem__(self, keys):
        if not isinstance(keys,(tuple,list)):
            keys = (keys, )
        if len(keys) != self.nargs:
            raise Exception( "Partial subscripting is not implemented yet." )
        s = set([type(k) for k in keys])
        if (int not in s) or len(s) > 1:
            raise Exception( "Only integers are allowed for subscripting." )
        newname = self.basename + '_' + '_'.join([str(k) for k in keys])
        if self.symbol_type == Variable:
            v = Variable(newname)
            return v
        elif self.symbol_type == Shock:
            return Shock(newname)
        elif self.symbol_type == Parameter:
            return Parameter(newname)


reg_rad = re.compile("([^_]+)")
reg_sep = re.compile("(&|_)")

def split_name_into_parts(a):
    s = a.replace('__','&')
    m = reg_rad.findall(a)
    rad = m[0]
    cont = m[1:]
    m = reg_sep.findall(s)
    exponents = []
    indices = []
    for i in range(len(cont)):
        if m[i] == '_':
          indices.append(cont[i])
        else:
          exponents.append(cont[i])
    return [rad, indices, exponents]

gl = ['alpha', 'beta', 'gamma', 'delta', 'eta','epsilon', 'iota', 'kappa',
'lambda', 'mu', 'nu', 'rho','pi', 'sigma', 'tau','theta','upsilon','omega','phi','psi','zeta', 'xi', 'chi',
'Gamma', 'Delta', 'Lambda', 'Sigma','Theta','Upsilon','Omega','Xi' , 'Pi' ,'Phi','Psi' ]
greek_letters = dict([ (x,'\\' + x ) for x in gl ])

def greekify(expr):
    if expr in greek_letters:
        return greek_letters[expr]
    else:
        return expr



class String(sympy.Basic):
    def __init__(self,name):
        super(sympy.Basic,self).__init__()
        self.name = name
    def __str__(self):
        return self.name

class ISymbol(sympy.Basic):


    def __init__(self,name,variable):
        super(sympy.Basic,self).__init__()
        self.name = name
        self.variable = variable
        
    def __getitem__(self,s):
        if isinstance(s,(String,str)):
            return sympy.Symbol(self.name + '_' + str(s))
        return ISymbol(self.name,s)

    def _eval_subs(self,a, b):
        if a == self.variable:
            return ISymbol(self.name,b)
        else:
            return self

class Sum(sympy.Basic):


    def __init__(self,index,set,expr):
        super(sympy.Basic,self).__init__()
        self.index = index
        self.set = set
        self.expr = expr

    def _eval_(self):
        return sum([self.expr.subs(self.index,s) for s in self.set])

class SVariant(sympy.Basic):


    def __init__(self,dummy,alternatives):
        super(sympy.Basic,self).__init__()
        self.dummy = dummy
        self.alternatives = dict(alternatives)

    def _eval_subs(self, a, b):
        a = str(a)
        b = str(b)
        if str(a) == str(self.dummy):
            b = str(b)
            altvs = [str(s) for s in self.alternatives]
            if b not in altvs:
                return self
            else:
                vv = self.alternatives[b]
                return vv
        else:
            return self


def Variant(dummy,alternatives):
    t = [(k,v) for k,v in alternatives.iteritems()]
    return SVariant( dummy, tuple(t))


def map_function_to_expression(f,expr):
    if expr.is_Atom:
        return( f(expr) )
    else:
        l = list( expr._args )
        args = []
        for a in l:
            args.append(map_function_to_expression(f,a))
        return( expr.__class__(* args) )


def timeshift(expr, tshift):
#    from dolo.symbolic.symbolic import TSymbol
    def fun(e):
        if isinstance(e,TSymbol):
            return e(tshift)
        else:
            return e
    return map_function_to_expression(fun, expr)


from sympy import __version__
from distutils.version import LooseVersion
#
# if LooseVersion(__version__) < LooseVersion('0.7.2'):
#
#     print('Importing old symbolics')
#     from dolo.symbolic.symbolic_old import TSymbol, Shock, Equation, Variable, Parameter
#
#     import warnings
#     warnings.warn('You have an old version of sympy (< 0.7.2). Support for older versions will be removed in later versions of dolo')
