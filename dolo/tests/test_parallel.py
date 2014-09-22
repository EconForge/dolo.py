import unittest

import time



def solve_something(name, args):

    from dolo import yaml_import
    from dolo.algos.fg.time_iteration import time_iteration as global_solve

    model = yaml_import('examples/models/{}.yaml'.format(name))
    dr = global_solve(model, **args)
    return dr

class TestParallel(unittest.TestCase):

    def test_solve_rbc_several_times(self):

        import pp

        job_server = pp.Server()

        tstart = time.time()

        jobs = []
        for i in range(10):
            f = job_server.submit(solve_something, ('rbc', {'smolyak_order':3}))
            jobs.append(f)

        t = [f() for f in jobs]

#        for dr in t:
#            print(dr.grid)

        tend = time.time()

        print('It took {} seconds with parallelization.'.format(tend - tstart))



    def test_solve_rbc_several_times_without_parallelization(self):

        tstart = time.time()

        jobs = []
        for i in range(10):
            f = solve_something('rbc', {'smolyak_order':3})
            jobs.append(f)

        #        for dr in t:
        #            print(dr.grid)

        tend = time.time()

        print('It took {} seconds without parallelization.'.format(tend - tstart))

