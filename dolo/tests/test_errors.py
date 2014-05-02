import unittest

class  ErrorsTestCase(unittest.TestCase):

    def test_omega_errors(self):

        from dolo import yaml_import
        from dolo.algos.time_iteration import time_iteration as global_solve

        model = yaml_import('examples/global_models/rbc.yaml')

        from dolo.algos.perturbations import approximate_controls

        dr = approximate_controls(model)
        dr_global = global_solve(model, smolyak_order=4, verbose=False, pert_order=1, method='newton', polish=True)


        sigma = model.covariances

        # cmodel = CModel(model)

        model.sigma = sigma

        s_0 = dr.S_bar

        from dolo.numeric.error_measures import  omega
        res = omega( dr, model, dr_global.bounds, [10,10], time_weight=[50, 0.96,s_0])


    def test_denhaan_errors(self):

        from dolo import yaml_import
        from dolo.algos.time_iteration import time_iteration as global_solve

        model = yaml_import('examples/global_models/rbc.yaml')

        from dolo.algos.perturbations import approximate_controls

        dr = approximate_controls(model)
        dr_global = global_solve(model, smolyak_order=4, verbose=False, pert_order=1, method='newton', polish=True)


        sigma = model.covariances

        model.sigma = sigma

        s_0 = dr.S_bar

        from dolo.numeric.error_measures import denhaanerrors

        [error_1, error_2] = denhaanerrors(model, dr, s_0)
        [error_1_glob, error_2_glob] = denhaanerrors(model, dr_global, s_0)

        print(error_1)
        print(error_1_glob)
        assert( max(error_1_glob) < 10-7) # errors with solyak colocations at order 4 are very small
        assert( max(error_2_glob) < 10-7)



        #print errs







if __name__ == '__main__':
    unittest.main()
