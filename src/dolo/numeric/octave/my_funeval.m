function [values] = my_funeval(points, derivs)

    global interp;
    coeffs = interp.coeffs;
    values = funeval(coeffs, interp.fspace, points, derivs);
