# this is a quick implementation
# some code should be shared with `model_numeric`

from dolo.compiler.model_numeric import NumericModel
import numpy


class DynareModel(NumericModel):

    def __compile_functions__(self):
        self.functions = self.__get_compiled_functions__()

    def __get_compiled_functions__(self, order=1):


        # constructs arguments of function f(y(1),y,y(-1),e,p)

        vv = self.symbols['variables']
        syms = [(v,1) for v in vv] + [(v,0) for v in vv] + [(v,-1) for v in vv]
        syms += [(s,0) for s in self.symbols['shocks']]
        params = self.symbols['parameters']

        # contsruct list of equations to differentiate
        # TODO: replace ugly splits, by call to equation parser:
        eqs = []
        for eq in self.symbolic.equations:
            m = str.split(eq,'=')
            if len(m) == 2:
                s = '{} - ({})'.format(m[1].strip(),m[0].strip())
                s = str.strip(s)
            else:
                s = eq
            eqs.append(s)

        from dolo.compiler.function_compiler_sympy import compile_higher_order_function


        f_dynamic = compile_higher_order_function(eqs, syms, params,
                                                order=order, funname='f_dynamic')

        e = self.calibration['shocks']
        f_static = lambda y, p: f_dynamic(numpy.concatenate([y,y,y,e]),p, order=0)

        functions = {
            'f_static': f_static,
            'f_dynamic': f_dynamic,
        }

        return functions





if __name__ == '__main__':

    from dolo import *

    fname = "/home/pablo/Programming/econforge/dolo/examples/models/rbc_dynare.yaml"


    smodel = yaml_import(fname, return_symbolic=True)

    infos = {
        'name' : 'anonymous',
        'type' : 'dynare'
    }

    model = DynareModel(smodel, infos=infos)


    f_static = model.functions['f_static']

    model.set_calibration(beta=0.95)
    p = model.calibration['parameters']
    y = model.calibration['variables']


    print(y)
    print(p)
    r = f_static(y,p)

    print("Residuals")
    print(r)

    model.set_calibration(beta=0.99)
    p = model.calibration['parameters']
    y = model.calibration['variables']


    print(y)
    print(p)
    r = f_static(y,p)

    print("Residuals")
    print(r)
