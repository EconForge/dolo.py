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

    ss = s0[None,:].repeat(N,axis=0)
    xx = x0[None,:].repeat(N,axis=0)
    ee = e0[None,:].repeat(N,axis=0)

    vec_res = f(ss,xx,ee,ss,xx,p)

    res = f(s0, x0, e0, s0, x0, p)

    assert(res.ndim==1)

    d = 0
    for i in range(N):
        d += abs(vec_res[i,:] - res).max()
    assert(d == 0)


def test_dtcscc__functions():

    # test a model defined without auxiliary variables
    from dolo import yaml_import
    model = yaml_import('examples/models/rbc_fg.yaml')

    s = model.calibration['states']
    x = model.calibration['controls']
    e = model.calibration['shocks']
    p = model.calibration['parameters']

    r = model.functions['arbitrage'](s,x,e,s,x,p)


def test_dtcscc_model():

    # test a model defined with auxiliary variables
    from dolo import yaml_import
    model = yaml_import('examples/models/rbc_full.yaml')

    s = model.calibration['states']
    x = model.calibration['controls']
    X = x

    y = model.calibration['auxiliaries']
    
    E = model.calibration['shocks']
    V = model.calibration['values']
    p = model.calibration['parameters']


    S = model.functions['transition'](s,x,E,p)
    r = model.functions['arbitrage'](s,x,E,S,X,p)
    y = model.functions['auxiliary'](s,x,p)
    v = model.functions['value'](s,x,S,X,V,p)

    z = model.functions['expectation'](S, X, p)

    x1 = model.functions['direct_response'](s,z, p)

    print('x1')
    print(x)
    print(x1)

    assert(abs(x-x1).max()<1e-12)


if __name__ == '__main__':
    test_dtcscc_model()
    test_dtcscc__functions()
    model_evaluation()
