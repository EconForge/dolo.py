import unittest

class StatesPerturbationsTestCase(unittest.TestCase):

    def test_first_order_accuracy(self):

        # This solves the optimal growth example at second order
        # and computes the second order correction to the steady-state
        # We test that both the statefree method and the perturbation to states
        # yield the same result.

        from dolo.misc.yamlfile import yaml_import
        model = yaml_import('examples/global_models/optimal_growth.yaml')


        from dolo.numeric.perturbations import solve_decision_rule
        from dolo.numeric.perturbations_to_states import approximate_controls


        coeffs = approximate_controls(model,order=2, return_dr=False)
        state_perturb = coeffs[0]

        dr = solve_decision_rule(model)
        statefree_perturb = dr['ys'] + dr['g_ss']/2.0
        ctls = model['variables_groups']['controls'] + model['variables_groups']['expectations']
        ctls_ind = [model.variables.index(v) for v in ctls]

        # the two methods should yield exactly the same result

        from numpy.testing import assert_almost_equal
        A = statefree_perturb[ctls_ind]
        B = state_perturb

        assert_almost_equal(A, B)  # we compare the risk-adjusted constants


    def test_perturbation_1(self):
        from dolo.misc.yamlfile import yaml_import
        model = yaml_import('examples/global_models/optimal_growth.yaml')
        from dolo.numeric.perturbations_to_states import approximate_controls
        dr = approximate_controls(model,order=1)


    def test_perturbation_2(self):
        from dolo.misc.yamlfile import yaml_import
        from dolo.numeric.perturbations_to_states import approximate_controls
        model = yaml_import('examples/global_models/optimal_growth.yaml')
        dr = approximate_controls(model,order=2)

    def test_perturbation_3(self):
        from dolo.misc.yamlfile import yaml_import
        from dolo.numeric.perturbations_to_states import approximate_controls
        model = yaml_import('examples/global_models/optimal_growth.yaml')
        dr = approximate_controls(model,order=3)

if __name__ == '__main__':
    unittest.main()
