import unittest

class  DivisionTestCase(unittest.TestCase):


    def test_division(self):

        import yaml

        with file('examples/global_models/rbc.yaml') as f:
            txt = f.read()

        yaml_content = yaml.load(txt)
        yaml_content['equations']['transition'][0] += ' + 1/2 - 0.5' # modify first transition equation
        yaml_content['calibration']['parameters']['alpha'] = '1/3'

        new_txt = yaml.dump( yaml_content )



        from dolo.misc.yamlfile import parse_yaml_text
        from dolo.symbolic.model import compute_residuals
        from dolo.symbolic.symbolic import Parameter

        model = parse_yaml_text(new_txt)

        [y,x,p] = model.read_calibration()

        i_alpha = model.parameters.index(Parameter('alpha'))
        assert( abs(p[i_alpha] - 1.0/3.0) < 0.00000001)

        res = compute_residuals(model)
        assert( abs(res['transition'][0]) < 0.00000001)



if __name__ == '__main__':
    unittest.main()
