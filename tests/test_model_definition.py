# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest

from dolo import *
#from dolo.misc.interactive import *

class  SyntaxTestCase(unittest.TestCase):

    def test_special_functions(self):
        set_variables('x')
        expr = log(x) * exp(x) * sin(x) * cos(x) * tan(x)

    def test_matrix_definitions(self):
        m = Matrix([[54,0.9],[90.3,0.1]])
        z = zeros(43)
        import sympy
        isinstance(m,sympy.Matrix)


class  DeclarationsTestCase(unittest.TestCase):

    def test_declare_variables(self):
        res = set_variables(["a"])
        assert(str(res) == "[a]")
        assert(str(a) == "a")
        assert(str(variables) == "[a]")
        add_variables('b')
        assert(str(res) == "[a, b]")
        add_variables('c d')
        assert(str(res) == "[a, b, c, d]")
        res = set_variables(["a","b"])
        assert(str(res) == "[a, b]")
        res = set_variables("a")
        assert(str(res) == "[a]")
        res = set_variables("a b")
        assert(str(res) == "[a, b]")

        res = set_shocks("a b")
        assert(str(res) == "[a, b]")
        assert(str(shocks) == "[a, b]")
        set_shocks("c")
        add_shocks("d e")
        assert(str(shocks) == "[c, d, e]")

        res = set_parameters("a b")
        assert(str(parameters) == "[a, b]")

if __name__ == '__main__':
    unittest.main()