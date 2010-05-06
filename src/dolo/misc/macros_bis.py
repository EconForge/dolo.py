from sympy import Basic,Symbol
from dolo import Variable
import re
import sympy

regex = re.compile('(@\{(\d+)\})')

class IVariable(Basic):

    def __new__(cls,name,dummies=None):
        resp = Basic.__new__(cls,name,dummies)
        resp.name = name
        if isinstance(dummies,(tuple,list)):
            resp._args = tuple(dummies)
            if len(dummies) == 0:
                return(Variable(name))
        elif dummies == None:
            resp._args = ()
        else:
            resp._args = (dummies,)
        return resp

    def _hashable_content(self):
        return (self.name, self.args)

    def nothing(self):
        return NotImplemented

    def __getitem__(self, keys):
        if not isinstance(keys,(tuple,list)):
            keys = (keys, )
        occurrences = regex.findall(self.name)
        nargs = len(occurrences)

        if len(keys) != len(occurrences):
            raise(Exception('Incorrect number of indices'))

        newname = self.name
        newargs = []
        for i in range(nargs):
            if isinstance(keys[i],(int,sympy.Integer)):
                newname = newname.replace(occurrences[i][0],str(keys[i]))
            else:
                newargs.append( keys[i])
                
        if newname == self.name:
            return self
        else:
            return( IVariable( newname, newargs ) )

    def _sympystr_(self):
        s = self.name
        occurrences = regex.findall(self.name)
        for i in range(len(self.args)):
            s = s.replace(occurrences[i][0],str(self.args[i]))
        return( s )

    def _eval_subs(self,old,new):
        if old == self:
            return new
        elif self._args == None:
            raise Exception("Substition impossible if no dummy variable is defined")
        else:
            newargs = list(self.args)
            for i in range(len(newargs)):
                if newargs[i] == old:
                    newargs[i] = new
            return ISymbol(self.name,tuple(newargs))
        
class Lag(Basic):
    def __new__(self,v,lag,**assumptions):
        if lag == 0:
            return v
        if isinstance(v,Lag):
            return Lag(v.args[0],lag + v.lag)
        if isinstance(v,TSymbol):
            return v(lag)
        else:
            self.lag = lag
            return Basic.__new__(self, v, lag, **assumptions)

    def _eval_subs(self,old,new):
        if old == self:
            return new
        v = self.args[0]
        lag = self.args[1]
        return Lag( v.subs(old,new), lag)

class h(Basic):
    def __new__(self,v,**assumptions):
        if isinstance(v,TSymbol):
            return v.S
        else:
            resp = Basic.__new__(self, v, **assumptions)
            return resp

    def _eval_subs(self,old,new):
        if old == self:
            return new
        v = self.args[0]
        return h( v.subs(old,new) )

##############################


sympy.var('i j')

s = IVariable('s')
print(s)
print s.args

q = IVariable('q_@{1}',(j))
print(q)
print q.name

qq = IVariable('qq_@{1}',(j))
print qq == q
print qq[1].name

t = IVariable('t_@{1}',(j))
print qq[i]

rr = IVariable('r_@{1}_@{2}',(i,j))
res =  rr[1,2]
print res.__class__

sympy.var('x y')
#
#eq = (qq + x + y)**2
#print eq.subs(qq,x)
#print eq.subs(j,j).subs(j,i).subs(i,x)
#
#print h(q)
#print Lag(rr,1)
#
#eqq = (rr + qq + h(q) + Lag(rr,1))
#print h(q).subs(i,1)
#print eqq
#print eqq.subs(i,1)
#print eqq.subs({i:1, j:2})



def qsum(expr,ind,range):
    s = 0
    for i in range:
        s += expr.subs(ind,i)
    return s
#
#print(qsum(eqq,i,range(3)))
#
