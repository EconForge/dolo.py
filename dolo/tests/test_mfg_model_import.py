import unittest


class test_mfg_import(unittest.TestCase):


    def test_model_import(self):

        from dolo import yaml_import
        fname = 'examples/models/rbc_mfga.yaml'
        model = yaml_import(fname)

    def test_model_print(self):

        from dolo import yaml_import
        fname = 'examples/models/rbc_mfga.yaml'
        model = yaml_import(fname)

        print(model)

    def test_markov_chain(self):

        from dolo import yaml_import
        fname = 'examples/models/rbc_mfga.yaml'
        model = yaml_import(fname)

        from dolo.numeric.discretization import multidimensional_discretization
        import numpy
        sigma = numpy.array([[0.01]])
        rho = 0.01
        [P,Q] = multidimensional_discretization(rho, sigma, 3)

        print(model.markov_chain)

    def test_options(self):

        from dolo import yaml_import
        fname = 'examples/models/rbc_mfga.yaml'
        model = yaml_import(fname)

        print( model.options )


if __name__ == '__main__':

    unittest.main()