def residuals(model, calib=None):

    if calib is None:
        calib = model.calibration

    y,e,p = model.calibration['variables', 'shocks', 'parameters']
    res = model.functions['f_static'](y, p)
    
    return {'dynare': res}
