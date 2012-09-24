from sympy import *
import re

regex = re.compile('@(.)')

#################
# general ideas #
#################
'''
Often,  the macrolanguage is used to iterate over equations
or to define  indexed operators. We can define corresponding structures in 
python mapping these operations.
They could be look like :

mfor(i,iset,expr) : represents a list of equations where i takes all
 values in iset in expr .
 
msum(i,iset,expr) : represents a sum of expressions indexed by i.

optionally we could add a condition like : msum(i,iset,cond=i<>j,expr)

When these objects are created they can be converted to Dynare code 
or to corresponding python objects (arrays of equations or expressions)

See below a quick code to demonstrate this.

There are two technical concerns to implement it :

- in dynare macrolanguage, we don't want to write loops like
mfor(i,[country1,country2,..., countryn],expr) when n is big

Instead we want to define first "countries = ..." and use the 
variable "countries" everywhere instead of the list
To implement this in python, we have to use a special object which will contain
the list of countries, as well as its own name.
The its definition can be outpouted somewhere in the macrolanguage code,
 while the loop output will contain the short name.
The syntax in python could ressemble :

countries = IndexSet('countries', ['france','australia'])
mfor(i,countries,expr)


- the substitution in expr is problematic. Without changing anything, we can simply
write :

Symbols('x v')
expr = x+v**2
mfor(v, ['country1','country2'],)

so that v will be replaced by the country name. But it would be clearer to
allow for variable subscripts in order to write :

IndexedVariable('country')
Symbols('x c')
expr = x+gdp[c]**2
mfor(c, ['country1','country2'], expr)

A good way to do it would be to define a new formal object which is a 
indexed variable. We show an example below.

Another type of substitution can occur in lags declaration, for instance we could
define a moving average by :
epxr = msum(i, [0,1,2,3], rho[i] * x(-i) ) where x is a variable

Again a simple way would be to define a formal object x(v) where is not an integer.

'''

#####################
# class definitions #
#####################

class mfor:
    '''
    This class implement the basic structure of an abstract loop
    '''
    
    dummy = None
    index = None
    expression = None
    
    def __init__(self,dummy,index,expression):
        self.dummy = dummy
        self.index = index
        self.expression = expression
        
    def eval(self):
        '''
        returns a list of expressions where dummy has been
        replaced by all elements of index
        '''
        return [self.expression.subs(self.dummy,i) for i in self.index]
    
    def to_dynare_string(self):
        txt = "@# for %s in %s\n" %(self.dummy,self.index)
        txt += str(self.expression.subs(self.dummy,Symbol("@{%s}" %str(self.dummy)))) + ";\n"
        txt += "@# endfor"
        return(txt)

class ISymbol(Basic):
    '''
    This class defines indexed symbols
    If country is an indexed symbol with index ['france','australia']
    the basic idea is that country['france'] or country['australia'] 
    will return an atomic symbol while country[v] with v being anything else
    returns an indexed symbol whose dummy variable is v.
    '''

    def __init__(self,name,index):
        super(ISymbol,self).__init__()
        self.name = name
        self.args = regex.findall(name)
        if 't' in dummy_vars:
            args.remove(t)
            if len(args) > 1:
                raise Exception('Multidimensional indexing not implemented')
            elif len(args) == 0:
                dummy = Symbol('i')
            else:
                dummy = Symbol(dummy_vars[0])
        self.index = tuple([sympify(i) for i in index])
        return None
    
    def set_index(self,index):
        self.index = [i for i in index]
        return self
        
    def __getitem__(self,key):
        skey = sympify(key)
        if skey in self.index:
        #if key in self.index:
            print('key in index')
            return Symbol(self.name + "_" + str(key) )
        else:
            print('key not in index')
            newname = self.name.replace(str(dummy),str(key))
            resp = ISymbol(newname,self.index)
            return resp
    
    def tostr(self,level=0):
        return(self.name + '_%s' %self.dummy)    

    def subs(self,key,value):
        return(self._eval_subs(key, value))
    
    def _eval_subs(self,key,value):
        print('subs called')
        if key == self.dummy:
            print 'good key'
            return self[value]
        else:
            print 'bad key'
            return self

###########
# Example #
###########
#
## this is a dummy variable
#v = Symbol('v')
#
## this is a regular symbol
#x = Symbol('x')
#print("\nFirst example :\n")
## it's enough to define a small loop which can be converted to python or Dynare language
#loop1 = mfor(v, ['y1','y2'],  x**2 + v * x )
#print("Python array : \n" + str( loop1.eval() ))
#print("Dynare string with macro instructions : \n" + loop1.to_dynare_string() )
#
#print("\n\nSecond example :\n")
## Let define an indexed symbol
## defining the indexing set is somewhat redundant with the definition
## of dummy variables but it makes things clearer
#country = ISymbol('country',('france','australia','newzealand'),x)
#
## now expressions can contain this indexed symbol
#expr = country[v]**2 -1
#print("Expression with a dummy variable : " + str(expr))
#print("The dummy variable is susbstituted by 'france' : " + str( expr.subs(v,'france') ) )
#
## define a loop
#loop = mfor(v,['france','australia'],expr)
#
## in python, this loop is evaluated to an array of expressions
#print("Python array : \n" + str( loop.eval() ))
#
## we can also get a Dynare string which uses the macrolanguage
#print("Dynare string with macro instructions : \n" + str( loop.to_dynare_string() ) )
