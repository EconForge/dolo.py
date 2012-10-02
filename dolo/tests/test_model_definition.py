# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest

from dolo import *
from dolo.misc.symbolic_interactive import def_variables, def_parameters, def_shocks
from sympy import *


class  SyntaxTestCase(unittest.TestCase):

    def test_special_functions(self):
        def_variables('x')
        expr = log(x) * exp(x) * sin(x) * cos(x) * tan(x)

    def test_matrix_definitions(self):
        m = Matrix([[54,0.9],[90.3,0.1]])
        z = zeros(43)
        import sympy
        isinstance(m,sympy.Matrix)


#class  DeclarationsTestCase(unittest.TestCase):
#
#    def test_declare_variables(self):
#        res = def_variables(["a"])
#        assert(str(res) == "[a]")
#        assert(str(a) == "a")
#        assert(str(variables) == "[a]")
#        def_variables('b')
#        assert(str(res) == "[a, b]")
#        def_variables('c d')
#        assert(str(res) == "[a, b, c, d]")
#        res = def_variables(["a","b"])
#        assert(str(res) == "[a, b]")
#        res = def_variables("a")
#        assert(str(res) == "[a]")
#        res = def_variables("a b")
#        assert(str(res) == "[a, b]")
#
#        res = set_shocks("a b")
#        assert(str(res) == "[a, b]")
#        assert(str(shocks) == "[a, b]")
#        set_shocks("c")
#        add_shocks("d e")
#        assert(str(shocks) == "[c, d, e]")
#
#        res = set_parameters("a b")
#        assert(str(parameters) == "[a, b]")

if __name__ == '__main__':
    unittest.main()