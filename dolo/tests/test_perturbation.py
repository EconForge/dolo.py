import unittest

class TestPerturbationCase(unittest.TestCase):

    def test_modfile_solve(self):

        from dolo import solve_decision_rule, dynare_import

        model = dynare_import('examples/dynare_modfiles/example1.mod')

        dr = solve_decision_rule(model, order=1)

        # at first order the decision rule is such that:
        # y_{t} - ybar = A (y_{t-1} - ybar) + B e_t

        print(dr['ys'])
        print(dr['g_a'])
        print(dr['g_e'])

        # it can be compared directly to dynare's decision rules
        # warning: Dynare's ordering is special
        print(dr.ghx)
        print(dr.ghu)

if __name__ == '__main__':

    unittest.main()