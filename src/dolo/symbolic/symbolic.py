
import sympy

from xml.etree import ElementTree as ET
import copy
import re


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
        else:
            self._assumptions['date'] = args['date']
        return(None)

    def __call__(self, shift):
        current_date = self.assumptions0['date']
        # we forget other assumptions
        v = type(self)
        return v( self.name, date = current_date + shift)


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

class Variable(TSymbol):
    """
    Define a new variable with : x = Variable('x')
    Then x(k) returns the value with lag k
    Internally x(-k) is a symbol with name x_bk
    """

    @property
    def S(self):
        v = Variable(self.name,date='S')
        return(v)

class Shock(TSymbol):
    """
    Define a new shock with : x = Shock('x')
    Then x(k) returns the value with la   g k
    """

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

# TODO : idea : subscripts are denoted by @
# upperscripts by $

class IndexedSymbol(sympy.Basic):

    def __init__(self, name, symbol_type):
        super(IndexedSymbol,self).__init__()
        self.name = name
        self.symbol_type = symbol_type


        self.nargs = name.count('@')
        self.basename = self.name.strip('@')
        if self.nargs == 0:
            raise Exception, "Indexed symbols must have at least one index."
        return(None)

    def __getitem__(self, keys):
        if not isinstance(keys,(tuple,list)):
            keys = (keys, )
        if len(keys) <> self.nargs:
            raise Exception, "Partial subscripting is not implemented yet."
        s = set([type(k) for k in keys])
        if (int not in s) or len(s) > 1:
            raise Exception, "Only integers are allowed for subscripting."
        newname = self.basename + '_' + '_'.join([str(k) for k in keys])
        if self.symbol_type == Variable:
            v = Variable(newname,0)
            return v
        elif self.symbol_type == Shock:
            return Shock(newname,0)
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
sympy.Symbol
from sympy.core.basic import Atom

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
        print s.__class__
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

if __name__ == '__main__':
    x = sympy.Symbol('x')
    i = sympy.Symbol('i')
    j = sympy.Symbol('j')
    c = sympy.Symbol('c')
    v = Variable('v',0)
    isym = ISymbol('test',x)
    print isym.is_Atom
    print isym[i]

    eq = (v + isym)**2
    #printq
    print eq
    print eq.subs(x,i)
    print isym._subs_old_new(x,i)
    print isym['paris']


    s = Sum(i,[0,5], x**i+1+isym[i])
    ss = Sum(c,[String('france'),'us'], isym[c])
    print s._eval_()
    print ss._eval_()
    print isym['france']
    print isym[String('france')]

    from sympy import Symbol
    dummy = Symbol


    t =  tuple()
    v = Variant( dummy, { i: 3, j: 43  } )
    v = Variant( dummy, { i: 3, j: 43  } )
    
    print v.subs(dummy,i)
