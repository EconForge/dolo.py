# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
from dolo.model.symbolic import Variable
from dolo.misc.latex_preview import latex


class  LatexPrintingTestCase(unittest.TestCase):

    def test_latex_printing(self):
        v = Variable('v')
        x = Variable('x')
        self.assertEqual(latex(v), 'v')
        self.assertEqual(latex(v(+1)), 'v(+1)')
        self.assertEqual(latex( x + v(+1) ), 'v(+1) + x')


if __name__ == '__main__':
    unittest.main()

