
def residuals(model, calib=None):

    if calib is None:
        calib = model.calibration

    from collections import OrderedDict
    res = OrderedDict()

    m = calib['exogenous']
    s = calib['states']
    x = calib['controls']
    p = calib['parameters']
    f = model.functions['arbitrage']
    g = model.functions['transition']

    res['transition'] = g(m,s,x,m,p)-s
    res['arbitrage'] = f(m,s,x,m,s,x,p)

    if 'value' in model.functions:
        v = calib['values']
        vfun = model.functions['value']
        res['value'] = vfun(m,s,x,v,m,s,x,v,p) - v

    return res
