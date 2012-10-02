# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
from dolo.symbolic.symbolic import TSymbol, Parameter, Variable

class  FormalCalculusTestCase(unittest.TestCase):

    def test_classes(self):
        from sympy import Symbol
        v = Variable('v')
        p = Parameter('p')
        s = Variable('v')
        symbs = [v,p,s]
        for i in symbs:
            assert(isinstance(i,Symbol))
        assert(isinstance(v,Variable))
        assert(isinstance(p,Parameter))
        assert(isinstance(s,Symbol))


    def test_forward_backward(self):
        "Tests whether lags operator are consistent"
        x = Variable('x')
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
        y = Variable('x')
        assert(y == x)
        #

    def test_derivative(self):
        y = Variable('y')
        z = Variable('z')
        eq = y + 3*y(1)
        assert( eq.diff(y) == 1)

    def test_present_value(self):
        # present operator should not depend on the initial lag
        x = Variable('x')
        assert( x(+1).P == x(+2).P )

    def test_steady_value(self):
        # special steady value is not present value
        x = Variable('x')
        assert( not (x.S) == x)
        assert( x(+1).S == x.S)

    def test_parameter(self):
        # we test initialization
        from sympy import Symbol
        p = Parameter('p')
        assert( isinstance(p,Parameter) )
        assert( isinstance(p,Symbol))
        assert( p.__latex__() == 'p')
        # greek letters are translated by sympy
        beta = Parameter('beta')
        delta = Parameter('delta')
        assert( beta.__latex__() == "\\beta")
        assert( delta.__latex__() == "\\delta")

    def test_variables_printing(self):
        v = Variable('v')
        vv = v(1)
        assert( str(vv) == 'v(1)' )
        assert( str(vv**2) == 'v(1)**2')

    def test_derivatives(self):
        x = TSymbol('x')
        xx = x(+1)
        eq = x + xx
        assert(eq.diff(x)==1)
#    def test_tvariable(self):
#        t = TSymbol('t')
#        s = TVariable('s')
#        print(s.__class__)
#        print(s(+1))
#        print(s(+2).__class__)

    def test_solving(self):
        import sympy
        x = TSymbol('x')
        y = TSymbol('y')
        z = TSymbol('z')
        eq = [ x + y - 1, x + 2*y -5 , z - x - 1]
        res = sympy.solve(eq,x,y,z)

if __name__ == '__main__':
    unittest.main()
