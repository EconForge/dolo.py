from dolo.compiler.model import Model
from typing import Dict, List
import numpy as np

from dolo.compiler.misc import CalibrationDict

def residuals(model:Model, calib=None)->Dict[str,List[float]]:

    if calib is None:
        calib = model.calibration

    res = dict()

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

def find_steady_state(model):

    n_s = len(model.calibration['states'])
    n_x = len(model.calibration['controls'])

    m = model.calibration['exogenous']
    p = model.calibration['parameters']

    def fobj(v):
        s = v[:n_s]
        x = v[n_s:]
        d = dict(states=s, controls=x, exogenous=m, parameters=p)
        res = residuals(model,d)
        return np.concatenate([res['transition'],res['arbitrage']])

    calib = model.calibration
    x0 = np.concatenate([calib['states'],calib['controls']])
    import scipy.optimize
    sol = scipy.optimize.root(fobj, x0)
    res = sol.x

    d = dict(exogenous=m, states=res[:n_s],controls=res[n_s:],parameters=p)
    return CalibrationDict(model.symbols, d)
