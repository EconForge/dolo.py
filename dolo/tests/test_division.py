import unittest

class  DivisionTestCase(unittest.TestCase):


    def test_division(self):

        import yaml

        with open('examples/models/rbc.yaml') as f:
            txt = f.read()

        yaml_content = yaml.load(txt)
        yaml_content['equations']['transition'][0] += ' + 1/2 - 0.5' # modify first transition equation
        yaml_content['calibration']['alpha'] = '1/3'

        txt = yaml.dump(yaml_content)

        from dolo import yaml_import
        print(txt)

        model = yaml_import('nofile', txt=txt)

        alpha = model.get_calibration('alpha')

        print(alpha)

        assert( abs(alpha - 1.0/3.0) < 0.00000001)

        res = model.residuals()
        print( abs(res['transition'][0] ) )
        assert( abs(res['transition'][0]) < 0.00000001)



if __name__ == '__main__':
    unittest.main()
