import numpy
from numpy import array, zeros


def calibration_to_vector(symbols, calibration_dict):

    from dolang.triangular_solver import solve_triangular_system
    from numpy import nan

    sol = solve_triangular_system(calibration_dict)

    calibration = dict()
    for group in symbols:
        # t = numpy.array([sol.get(s, nan) for s in symbols[group]], dtype=float)
        t = numpy.array([sol.get(s, nan) for s in symbols[group]], dtype=float)
        calibration[group] = t

    return calibration


def calibration_to_dict(symbols, calib):

    if not isinstance(symbols, dict):
        symbols = symbols.symbols

    d = dict()
    for group, values in calib.items():
        if group == "covariances":
            continue
        syms = symbols[group]
        for i, s in enumerate(syms):
            d[s] = values[i]

    return d


from dolo.compiler.misc import calibration_to_dict

import copy

equivalent_symbols = dict(actions="controls")


class LoosyDict(dict):
    def __init__(self, **kwargs):

        kwargs = kwargs.copy()
        if "equivalences" in kwargs.keys():
            self.__equivalences__ = kwargs.pop("equivalences")
        else:
            self.__equivalences__ = dict()
        super().__init__(**kwargs)

    def __getitem__(self, p):

        if p in self.__equivalences__.keys():
            k = self.__equivalences__[p]
        else:
            k = p
        return super().__getitem__(k)


class CalibrationDict:

    # cb = CalibrationDict(symbols, calib)
    # calib['states'] -> array([ 1.        ,  9.35497829])
    # calib['states','controls'] - > [array([ 1.        ,  9.35497829]), array([ 0.23387446,  0.33      ])]
    # calib['z'] - > 1.0
    # calib['z','y'] -> [1.0, 0.99505814380953039]

    def __init__(self, symbols, calib, equivalences=equivalent_symbols):
        calib = copy.deepcopy(calib)
        for v in calib.values():
            v.setflags(write=False)
        self.symbols = symbols
        self.flat = calibration_to_dict(symbols, calib)
        self.grouped = LoosyDict(
            **{k: v for (k, v) in calib.items()}, equivalences=equivalences
        )

    def __getitem__(self, p):
        if isinstance(p, tuple):
            return [self[e] for e in p]
        if p in self.symbols.keys() or (p in self.grouped.__equivalences__.keys()):
            return self.grouped[p]
        else:
            return self.flat[p]


def allocating_function(inplace_function, size_output):
    def new_function(*args, **kwargs):
        val = numpy.zeros(size_output)
        nargs = args + (val,)
        inplace_function(*nargs)
        if "diff" in kwargs:
            return numdiff(new_function, args)
        return val

    return new_function


def numdiff(fun, args):
    """Vectorized numerical differentiation"""

    # vectorized version

    epsilon = 1e-8
    args = list(args)
    v0 = fun(*args)
    N = v0.shape[0]
    l_v = len(v0)
    dvs = []
    for i, a in enumerate(args):
        l_a = (a).shape[1]
        dv = numpy.zeros((N, l_v, l_a))
        nargs = list(args)  # .copy()
        for j in range(l_a):
            xx = args[i].copy()
            xx[:, j] += epsilon
            nargs[i] = xx
            dv[:, :, j] = (fun(*nargs) - v0) / epsilon
        dvs.append(dv)
    return [v0] + dvs
