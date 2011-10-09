import unittest

class StatesPerturbationsTestCase(unittest.TestCase):

    def test_perturbation(self):

        # This solves the optimal growth example at second order
        # and computes the second order correction to the steady-state
        # We test that both the statefree method and the perturbation to states
        # yield the same result.

        from dolo.misc.yamlfile import yaml_import
        model = yaml_import('../examples/global_models/optimal_growth.yaml')

        from dolo.numeric.perturbations_to_states import approximate_controls

        [Xbar, [X_s,X_ss],[X_tt]]  = approximate_controls(model,2)
        state_perturb = (Xbar + X_tt/2.0)


        from dolo.numeric.perturbations import solve_decision_rule
        dr = solve_decision_rule(model)
        statefree_perturb = dr['ys'] + dr['g_ss']/2.0
        ctls = model['variables_groups']['controls'] + model['variables_groups']['expectations']
        ctls_ind = [model.variables.index(v) for v in ctls]

        # the two methods should yield exactly the same result

        from numpy.testing import assert_almost_equal
        A = statefree_perturb[ctls_ind]
        B = state_perturb

        assert_almost_equal(A, B)

    def test_higher_order_perturbation(self):

        # This solves the optimal growth example at second order
        # and computes the second order correction to the steady-state
        # We test that both the statefree method and the perturbation to states
        # yield the same result.

        from dolo.misc.yamlfile import yaml_import
        model = yaml_import('../examples/global_models/optimal_growth.yaml')

        from dolo.numeric.perturbations_to_states import approximate_controls

        [Xbar,X_s,X_ss,X_sss]  = approximate_controls(model,3)

if __name__ == '__main__':
    unittest.main()