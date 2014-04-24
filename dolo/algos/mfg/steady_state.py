

def residuals(model, calib=None):

    if calib is None:
        calib = model.calibration

    from collections import OrderedDict
    res = OrderedDict()

    if model.model_type == "mfg":

        m = calib['markov_states']
        s = calib['states']
        x = calib['controls']
        p = calib['parameters']
        f = model.functions['arbitrage']
        g = model.functions['transition']

        res['transition'] = g(m,s,x,m,p)-s
        res['arbitrage'] = f(m,s,x,m,s,x,p)


    elif model.model_type  == "mfga":

        m = calib['markov_states']

        s = calib['states']
        x = calib['controls']
        y = calib['auxiliaries']
        p = calib['parameters']

        f = model.functions['arbitrage']
        g = model.functions['transition']
        a = model.functions['auxiliary']

        res['transition'] = g(m,s,x,y,m,p)-s
        res['arbitrage'] = f(m,s,x,y,m,s,x,y,p)
        res['auxiliary'] = a(m,s,x,p)-y

    else:
        raise Exception("Not implemented")

    return res
