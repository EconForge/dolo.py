
def test_omega_errors():

        from dolo import yaml_import
        from dolo.algos.dtcscc.time_iteration import time_iteration as time_iteration

        model = yaml_import('examples/models/rbc.yaml')

        from dolo.algos.dtcscc.perturbations import approximate_controls

        dr = approximate_controls(model)
        dr_global = time_iteration(model, smolyak_order=3, verbose=False, pert_order=1)

        sigma = model.covariances

        model.sigma = sigma

        s_0 = dr.S_bar

        from dolo.algos.dtcscc.accuracy import  omega

        res_1 = omega( model, dr, orders=[10,10], time_discount=0.96)
        res_2 = omega( model, dr_global)

        print(res_1)
        print(res_2)


def test_denhaan_errors():

        from dolo import yaml_import
        from dolo.algos.dtcscc.time_iteration import time_iteration as time_iteration

        model = yaml_import('examples/models/rbc.yaml')

        from dolo.algos.dtcscc.perturbations import approximate_controls

        dr = approximate_controls(model)

        dr_global = time_iteration(model, interp_type='smolyak', smolyak_order=4, verbose=False)

        s0 = model.calibration['states']
        sigma = model.covariances

        model.sigma = sigma

        from dolo.algos.dtcscc.accuracy import denhaanerrors

        denerr_1 = denhaanerrors(model, dr)
        denerr_2 = denhaanerrors(model, dr_global)

        print(denerr_1)
        print(denerr_2)
        print(denerr_2['max_errors'][0])


        assert( max(denerr_2['max_errors']) < 10-7) # errors with solyak colocations at order 4 are very small
        # assert( max(denerr_1['mean_errors']) < 10-7)










if __name__ == '__main__':

    test_denhaan_errors()
    test_omega_errors()
