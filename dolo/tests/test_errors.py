import unittest

class  ErrorsTestCase(unittest.TestCase):

    def test_denhaan_errors(self):

        from dolo.misc.yamlfile import yaml_import
        from dolo.numeric.global_solve import global_solve

        model = yaml_import('examples/global_models/rbc.yaml')


        from dolo.compiler.compiler_global import CModel
        from dolo.numeric.perturbations_to_states import approximate_controls

        dr = approximate_controls(model, solve_systems=True)
        dr_global = global_solve(model, smolyak_order=4, verbose=False, pert_order=1)


        sigma = model.read_covariances()

        cmodel = CModel(model, substitute_auxiliary=True, solve_systems=True)
        cmodel.sigma = sigma

        s_0 = dr.S_bar

        from dolo.numeric.error_measures import denhaanerrors

        [error_1, error_2, sim, err] = denhaanerrors(cmodel, dr, s_0)
        [error_1_glob, error_2_glob, sim_glob, err_glob] = denhaanerrors(cmodel, dr_global, s_0)

        assert( max(error_1_glob) < 10-7) # errors with solyak colocations at order 4 are very small
        assert( max(error_2_glob) < 10-7)



        #print errs







if __name__ == '__main__':
    unittest.main()
