import warnings
from collections import OrderedDict

import numpy
from scipy.optimize import root

from dolo.numeric.misc import MyJacobian


def find_deterministic_equilibrium(model, constraints=None,
                                   return_jacobian=False):
    '''
    Finds the steady state calibration.

    Taking the value of parameters as given, finds the values for endogenous
    variables consistent with the deterministic steady-state.

    This function requires the specification of the first order equations.

    Parameters
    ----------
    model: NumericModel
        an `(f,g)` compliant model
    constraints: dict
        a dictionaries with forced values.
        Use it to set shocks to non-zero values or to add additional
        constraints in order to avoid unit roots.

    Returns:
    --------
    OrderedDict:
        calibration dictionary (i.e. endogenous variables and parameters by
        type)
    '''

    f = model.functions['arbitrage']
    g = model.functions['transition']

    s0 = model.calibration['states']
    x0 = model.calibration['controls']
    p = model.calibration['parameters']

    if 'shocks' in model.calibration:
        e0 = model.calibration['shocks'].copy()
    else:
        e0 = numpy.zeros(len(model.symbols['shocks']))

    n_e = len(e0)

    z = numpy.concatenate([s0, x0, e0])

    symbs = model.symbols['states'] + model.symbols['controls']
    addcons_ind = []
    addcons_val = []

    if constraints is None:
        constraints = dict()

    for k in constraints:
        if k in symbs:
            i = symbs.index(k)
            addcons_ind.append(i)
            addcons_val.append(constraints[k])
        elif k in model.symbols['shocks']:
            i = model.symbols['shocks'].index(k)
            e0[i] = constraints[k]
        else:
            raise Exception(
                "Invalid symbol '{}' for steady_state constraint".format(k))

    def fobj(z):
        s = z[:len(s0)]
        x = z[len(s0):-n_e]
        e = z[-n_e:]

        S = g(s, x, e, p)
        r = f(s, x, e, s, x, p)
        d_e = e - e0
        d_sx = z[addcons_ind] - addcons_val
        res = numpy.concatenate([S - s, r, d_e, d_sx])
        return res

    jac = MyJacobian(fobj)(z)

    if return_jacobian:
        return jac

    rank = numpy.linalg.matrix_rank(jac)

    if rank < len(z):
        msg = """\
        There are {} equilibrium variables to find, but the jacobian \
        matrix is only of rank {}. The solution is indeterminate."""
        warnings.warn(msg.format(len(z), rank))

    sol = root(fobj, z, method='lm')
    steady_state = sol.x

    s = steady_state[:len(s0)]
    x = steady_state[len(s0):-n_e]
    e = steady_state[-n_e:]

    calib = OrderedDict(states=s, controls=x, shocks=e, parameters=p.copy())

    if 'auxiliary' in model.functions:
        a = model.functions['auxiliary'](s, x, p)
        calib['auxiliaries'] = a

    from dolo.compiler.misc import CalibrationDict
    return CalibrationDict(model.symbols, calib)

def residuals(model, calib=None):
    '''
    Computes the residuals associated to a calibration.

    Parameters
    ----------
    model: NumericModel

    calib: OrderedDict
        calibration dictionary (i.e. endogenous variables and parameters by
        type)

    Returns:
    --------
    OrderedDict:
        residuals vectors by equation type
    '''

    if calib is None:
        calib = model.calibration

    from collections import OrderedDict
    res = OrderedDict()

    if 'auxiliaries' not in model.symbols:

        s = calib['states']
        x = calib['controls']
        e = calib['shocks']
        p = calib['parameters']
        f = model.functions['arbitrage']
        g = model.functions['transition']

        res['transition'] = g(s, x, e, p) - s
        res['arbitrage'] = f(s, x, e, s, x, p)
    else:

        s = calib['states']
        x = calib['controls']
        y = calib['auxiliaries']
        e = calib['shocks']
        p = calib['parameters']

        f = model.functions['arbitrage']
        g = model.functions['transition']
        a = model.functions['auxiliary']

        res['transition'] = g(s, x, e, p) - s
        res['arbitrage'] = f(s, x, e, s, x, p)
        res['auxiliary'] = a(s, x, p) - y

        if 'value' in model.functions:
            rew = model.functions['value']
            v = calib['values']
            res['value'] = rew(s, x, s, x, v, p) - v

    return res
#
# def find_steady_state(model, e=None, force_states=None, constraints=None, return_jacobian=False):
#     '''n
#     Finds the steady state corresponding to exogenous shocks :math:`e`.
#
#     :param model: an "fg" model.
#     :param e: a vector with the value for the exogenous shocks.
#     :param force_values: (optional) a vector where finite values override the equilibrium conditions. For instance a vector :math:`[0,nan,nan]` would impose that the first state must be equal to 0, while the two next ones, will be determined by the model equations. This is useful, when the deterministic model has a unit root.
#     :return: a list containing a vector for the steady-states and the corresponding steady controls.
#     '''
#
#     s0 = model.calibration['states']
#     x0 = model.calibration['controls']
#     p = model.calibration['parameters']
#     z = numpy.concatenate([s0, x0])
#
#     if e is None:
#         e = numpy.zeros( len(model.symbols['shocks']) )
#     else:
#         e = e.ravel()
#
#     if constraints is not None:
#         if isinstance(constraints, (list, tuple)):
#             inds =  numpy.where( numpy.isfinite( force_values ) )[0]
#             vals = force_values[inds]
#         elif isinstance(constraints, dict):
#             inds = [model.symbols['states'].index(k) for k in force_values.keys()]
#             vals = force_values.values()
#
#     def fobj(z):
#         s = z[:len(s0)]
#         x = z[len(s0):]
#         S = model.functions['transition'](s,x,e,p)
#         r = model.functions['arbitrage'](s,x,s,x,p)
#         res = numpy.concatenate([S-s, r,  ])
#         if force_values is not None:
#             add = S[inds]-vals
#             res = numpy.concatenate([res, add])
#         return res
#
#     from trash.dolo.numeric.solver import MyJacobian
#     jac = MyJacobian(fobj)( z )
#     if return_jacobian:
#         return jac
#
#
#     rank = numpy.linalg.matrix_rank(jac)
#     if rank < len(z):
#         import warnings
#         warnings.warn("There are {} equilibrium variables to find, but the jacobian matrix is only of rank {}. The solution is indeterminate.".format(len(z),rank))
#
#     from scipy.optimize import root
#     sol = root(fobj, z, method='lm')
#     steady_state = sol.x
#
#     return [steady_state[:len(s0)], steady_state[len(s0):]]
#

if __name__ == '__main__':

    from dolo import *
    from numpy import nan

    from dolo.algos.steady_state import find_steady_state

    model = yaml_import("examples/models/open_economy.yaml")

    ss = find_steady_state(model)

    print("Steady-state variables")
    print("states: {}".format(ss[0]))
    print("controls: {}".format(ss[1]))

    jac = find_steady_state(model, return_jacobian=True)

    rank = numpy.linalg.matrix_rank(jac)

    sol2 = find_steady_state(
        model,
        force_values=[0.3, nan]
    )  # -> returns steady-state, using calibrated values as starting point
    sol3 = find_steady_state(
        model,
        force_values={'W_1': 0.3}
    )  # -> returns steady-state, using calibrated values as starting point

    print(sol2)
    print(sol3)

#    steady_state( model, e ) # -> sets exogenous values for shocks

#    steady_state( model, {'e_a':1, 'e':9}, {'k':[8,9]})
