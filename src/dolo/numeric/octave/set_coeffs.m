function [coeffs] = set_coeffs(values)

    global interp;

    coeffs = funfitxy( interp.fspace, interp.Phi, values );
    interp.coeffs=coeffs;
