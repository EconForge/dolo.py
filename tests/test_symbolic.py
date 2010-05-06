# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
from dolo.model.symbolic import Variable,Parameter,TSymbol
import sympy

class  FormalCalculusTestCase(unittest.TestCase):

    def test_classes(self):
        from sympy import Symbol
        v = Variable('v',0)
        p = Parameter('p',0)
        s = Variable('v',0)
        symbs = [v,p,s]
        for i in symbs:
            assert(isinstance(i,Symbol))
        assert(isinstance(v,Variable))
        assert(isinstance(p,Parameter))
        assert(isinstance(s,Symbol))


    def test_forward_backward(self):
        "Tests whether lags operator are consistent"
        x = Variable('x',0)
        x1 = x(+1)
        x_1 = x(-1)
        z = x_1(+2)
        assert(x.lag == 0)
        assert(x1.lag == 1)
        assert(x_1.lag == -1)
        assert(z.lag == 1)
        assert(x1 == z)
        assert(x1.P == x)
        # maybe we could check that both variables refer to the same object
        #ssert(id(x1) == id(z) )
        # x and y denote the same formal variables but are in two different instances
        y = Variable('x',0)
        assert(y == x)

    def test_present_value(self):
        # present operator should not depend on the initial lag
        x = Variable('x',0)
        assert( x(+1).P == x(+2).P )

    def test_steady_value(self):
        # special steady value is not present value
        x = Variable('x',0)
        assert( not (x.S) == x)
        assert( x(+1).S == x.S)

    def test_parameter(self):
        # we test initialization
        from sympy import Symbol
        p = Parameter('p',0)
        assert( isinstance(p,Parameter) )
        assert( isinstance(p,Symbol))
        assert( p.__latex__() == 'p')
        # greek letters are translated by sympy
        beta = Parameter('beta',0)
        delta = Parameter('delta',0,latex_name='d')
        assert( beta.__latex__() == "\\beta")
        assert( delta.__latex__() == "d")

    def test_derivatives(self):
        x = TSymbol('x',0)
        xx = x(+1)
        eq = x + xx
        print eq.diff(x)
        print(x == xx)
        assert(eq.diff(x)==1)
#    def test_tvariable(self):
#        t = TSymbol('t')
#        s = TVariable('s')
#        print(s.__class__)
#        print(s(+1))
#        print(s(+2).__class__)

    def test_solving(self):
        x = TSymbol('x',0)
        y = TSymbol('y',0)
        z = TSymbol('z',0)
        eq = [ x + y - 1, x + 2*y -5 , z - x - 1]
        res = sympy.solve(eq,x,y,z)

if __name__ == '__main__':
    unittest.main()

