import unittest
import numpy



def test_web_import():

    from dolo import yaml_import


    model = yaml_import("https://raw.githubusercontent.com/EconForge/dolo/master/examples/models/rbc.yaml")
    assert(len(model.symbols['states'])==2)

def model_evaluation(compiler='numpy', data_layout='columns'):

    from dolo import yaml_import

    #model = yaml_import('examples/models/rbc_fg.yaml', compiler=compiler, order=data_layout)
    model = yaml_import('examples/models/rbc_fg.yaml')

    s0 = model.calibration['states']
    x0 = model.calibration['controls']
    e0 = model.calibration['shocks']
    p = model.calibration['parameters']

    assert(s0.ndim == 1)
    assert(x0.ndim == 1)
    assert(p.ndim == 1)

    N = 10
    f = model.functions['arbitrage']

    if data_layout == 'rows':
        ss = numpy.ascontiguousarray( numpy.tile(s0, (N,1)).T )
        xx = numpy.ascontiguousarray( numpy.tile(x0, (N,1)).T )
        ee = e0[:,None].repeat(N,axis=1)
    else:
        ss = ( numpy.tile(s0, (N,1)) )
        xx = ( numpy.tile(x0, (N,1)) )
        ee = e0[None,:].repeat(N,axis=0)

    vec_res = f(ss,xx,ee,ss,xx,p)

    res = f(s0, x0, e0, s0, x0, p)

    assert(res.ndim==1)

    d = 0
    for i in range(N):
        if data_layout == 'columns':
            d += abs(vec_res[i,:] - res).max()
        else:
            d += abs(vec_res[:,i] - res).max()
    assert(d == 0)


def test_fg_functions():

    from dolo import yaml_import
    model = yaml_import('examples/models/rbc_fg.yaml')

    s = model.calibration['states']
    x = model.calibration['controls']
    e = model.calibration['shocks']
    p = model.calibration['parameters']

    r = model.functions['arbitrage'](s,x,e,s,x,p)


def test_fga_model():

    from dolo import yaml_import
    model = yaml_import('examples/models/rbc.yaml')

    s = model.calibration['states']
    x = model.calibration['controls']
    y = model.calibration['auxiliaries']
    e = model.calibration['shocks']
    v = model.calibration['values']
    p = model.calibration['parameters']


    r = model.functions['arbitrage'](s,x,e,s,x,p)
    S = model.functions['transition'](s,x,e,p)
    y = model.functions['auxiliary'](s,x,p)
    V = model.functions['value'](s,x,s,x,v,p)



class TestModelImport(unittest.TestCase):

    # def test_standard_import_rows(self):
    #
    #     model_evaluation(data_layout='rows')
    #
    # def test_import_numexpr_rows(self):
    #
    #     model_evaluation(compiler='numexpr', data_layout='rows')

    def test_standard_import_columns(self):

        model_evaluation(data_layout='columns')

    def test_import_numexpr_columns(self):

        model_evaluation(compiler='numexpr', data_layout='columns')

if __name__ == '__main__':
    unittest.main()
