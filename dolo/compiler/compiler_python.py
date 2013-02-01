
class GModel(object):
    '''Generic compiled model'''

    def __init__(self, model, model_type=None, recipes=None, compiler=None):

        # this part is actually common to all compilers

        if model_type is None:
            model_type = model['original_data']['model_type']

        self.model = model

        from dolo.symbolic.validator import validate
        if isinstance(model_type, str):
            from dolo.symbolic.recipes import recipes as recipes_dict
            if recipes is not None:
                recipes_dict.update(recipes)
            self.recipe = recipes_dict[model_type]
            validate(model, self.recipe)
        else:
            # if model_type is not a string, we assume it is a recipe (why ?)
            self.recipe = model_type

        self.model_type = self.recipe['model_type']

        self.__create_functions__(compiler)

    def __create_functions__(self, compiler):
        recipe = self.recipe

        model = self.model
        parms = model['parameters_ordering']

        functions = {}

        for eqg in self.model['equations_groups']:
            args = []
            is_a_definition = 'definition' in recipe['equation_type'][eqg]
            if is_a_definition:
                arg_specs = recipe['equation_type'][eqg]['rhs']
            else:
                arg_specs = recipe['equation_type'][eqg]
            arg_names = []
            for syms in arg_specs:
                [sgn,time] = syms
                if syms[0] == 'shocks':
                    args.append( [ s(time) for s in model['shocks_ordering'] ] )
                else:
                    args.append( [ s(time) for s in model['variables_groups'][sgn] ] )
                if time == 1:
                    stime = '_f'
                elif time == -1:
                    stime = '_p'
                else:
                    stime = ''
                arg_names.append( sgn + stime)

            equations = self.model['equations_groups'][eqg]

            if is_a_definition:
                from dolo.compiler.common import solve_recursive_block
                equations = solve_recursive_block(equations)
                equations = [eq.rhs for eq in equations]
            else:
                equations = [eq.gap for eq in equations]

            if compiler == 'numexpr':
                from dolo.compiler.function_compiler_numexpr import compile_multiargument_function
            elif compiler == 'theano':
                from dolo.compiler.function_compiler_theano import compile_multiargument_function
            else:
                from dolo.compiler.function_compiler import compile_multiargument_function

            functions[eqg] = compile_multiargument_function(equations, args, arg_names, parms, fname = eqg)


        calibration = model.calibration

        import numpy
        from collections import OrderedDict
        for k,v in calibration.iteritems():
            if isinstance(v, OrderedDict):
                for l in v:
                    v[l] = numpy.array(v[l], dtype=numpy.double)
            else:
                calibration[k] = numpy.array(calibration[k], dtype=numpy.double)

        symbols = {}
        for vn, vg in model['variables_groups'].iteritems():
            symbols[vn] = [str(v) for v in vg]
        symbols['shocks'] = [str(v) for v in model.shocks]
        symbols['parameters'] = [str(v) for v in model.parameters]

        self.calibration = calibration
        self.symbols = symbols
        self.functions = functions

if __name__ == '__main__':
    from dolo import *
    import numpy


    model = yaml_import('examples/global_models/rbc.yaml')

    gm = GModel(model, compiler='numexpr')
#    gm = GModel(model, compiler='theano')
#    gm = GModel(model)

    ss = gm.calibration['steady_state']['states']
    xx = gm.calibration['steady_state']['controls']
    aa = gm.calibration['steady_state']['auxiliary']
    p = gm.calibration['parameters']

    ee = numpy.array([0],dtype=numpy.double)

    N = 1000000

    ss = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ss).T, (1,N)) )
    xx = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(xx).T, (1,N)) )
    aa = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(aa).T, (1,N)) )
    ee = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ee).T, (1,N)) )


    g = gm.functions['transition']
    f = gm.functions['arbitrage']
    import time

    t1 = time.time()

    tmp = g(ss,xx,aa,ee,p)
    t2 = time.time()

    for i in range(100):
        tmp = f(ss,xx,aa,ss,xx,aa,p)
    t3 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t3-t2))

    print(gm)
