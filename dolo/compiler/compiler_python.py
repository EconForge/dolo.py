from dolo.misc.caching import memoized


class GModel(object):

    '''Generic compiled model object

    :param  model: (SModel)  a symbolic model
    :param  model_type: (str) model type (e.g. ``"fg"`` or ``"fga"``
    :param recipes: (dict) dictionary of recipes (must contain ``model_type`` as a key)
    :param compiler: (str) compiler to use. One of ``numpy``, ``numexpr``, ``theano``

    The class contains the following fields:

    :attr symbols: (dict) symbol groups -> list of symbol names defining each group
    :attr functions: (dict) equation names -> compiled functions
    :attr calibration: (dict) symbol groups -> vector of calibrated values for each group

    :attr model: (optional) link to the original symbolic model


    '''

    model = None
    calibration = None
    functions = None
    symbols = None

    @property
    def variables(self):
        vars = []
        for vg in self.symbols:
            if vg not in ('parameters','shocks'):
                vars.extend( self.symbols[vg] )
        return vars

    def __init__(self, model, model_type=None, recipes=None, compiler=None, order='rows'):

        # this part is actually common to all compilers

        if model_type is None:
            model_type = model.__data__['model_type']

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

        self.__create_functions__(compiler, order=order)

    def __create_functions__(self, compiler, order='rows'):
        recipe = self.recipe

        model = self.model
        parms = model.symbols_s['parameters']

        if compiler == 'numexpr':
            from dolo.compiler.function_compiler_numexpr import compile_multiargument_function
        elif compiler == 'theano':
            from dolo.compiler.function_compiler_theano import compile_multiargument_function
        elif compiler == 'numba':
            from dolo.compiler.function_compiler_numba import compile_multiargument_function
        elif compiler == 'numba_gpu':
            from dolo.compiler.function_compiler_numba_gpu import compile_multiargument_function
        else:
            from dolo.compiler.function_compiler import compile_multiargument_function


        functions = {}

        for eqg in self.model.equations_groups:
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
                    args.append( [ s(time) for s in model.symbols_s['shocks'] ] )
                else:
                    args.append( [ s(time) for s in model.symbols_s[sgn] ] )
                if time == 1:
                    stime = '_f'
                elif time == -1:
                    stime = '_p'
                else:
                    stime = ''
                arg_names.append( sgn + stime)

            equations = self.model.equations_groups[eqg]

            if is_a_definition:
                from dolo.compiler.common import solve_recursive_block
                equations = solve_recursive_block(equations)
                equations = [eq.rhs for eq in equations]
            else:
                equations = [eq.gap for eq in equations]


            functions[eqg] = compile_multiargument_function(equations, args, arg_names, parms, fname = eqg, order=order)


        self.__update_calibration__()

        from collections import OrderedDict
        l =  [ (vg, [str(s) for s in model.symbols_s[vg]] ) for vg in (recipe['variable_type'] + ['shocks','parameters']) ]
        symbols = OrderedDict( l )

        self.symbols = symbols
        self.functions = functions

    def __update_calibration__(self):
        import numpy
        from collections import OrderedDict
        calibration = self.model.calibration
        for k,v in calibration.iteritems():
            if isinstance(v, OrderedDict):
                for l in v:
                    v[l] = numpy.array(v[l], dtype=numpy.double)
            else:
                calibration[k] = numpy.array(calibration[k], dtype=numpy.double)
        self.calibration = calibration

    def set_calibration(self,*args):
        """Updates the model calibration while respecting dependences between parameters.
        :param args: either two parameters ``key``, ``value`` or a dictionary mapping several keys to several values
             each key must be a string among the symbols of the model
        """
        if len(args) == 2:
            d = {args[0]:args[1]}
        else:
            d = args[0]
        self.model.set_calibration(d)
        self.__update_calibration__()

    def get_calibration(self,name):
        """Get the calibrated value for one or several variables
        :param name: string or list of string with the parameter names to query
        :return: parameter(s) name(s)
        """

        is_iterable = isinstance( name, (list,tuple) )
        if is_iterable:
            return [self.get_calibration(n) for n in name]

        name = str(name)
        # get symbol group containing name
        group = [sg for sg in self.symbols if name in self.symbols[sg]]
        if len(group)==0:
            raise Exception('Symbol {} is not defined for this model'.format(name))
        assert(len(group)==1)
        group = group[0]

        ind = self.symbols[group].index(name)
        return self.calibration[group][ind]

    @property
    @memoized
    def x_bounds(self):

        model = self.model

        states = model.symbols_s['states']
        parameters = model.parameters
        from dolo.compiler.function_compiler import compile_multiargument_function

        [lower_bounds_symbolic, upper_bounds_symbolic] = self.model.get_complementarities()['arbitrage']
        lb = compile_multiargument_function( lower_bounds_symbolic, [states], ['s'], parameters, fname='lb')
        ub = compile_multiargument_function( upper_bounds_symbolic, [states], ['s'], parameters, fname='ub' )

        return [lb,ub]


if __name__ == '__main__':

    from dolo import *
    import numpy


    gm = yaml_import('examples/global_models/rbc.yaml', compiler='numpy')

    # print(model.__class__)
    # gm = GModel(model, compiler='numexpr')
    # # gm = GModel(model, compiler='theano')
#    gm = GModel(model)

    ss = gm.calibration['states']
    xx = gm.calibration['controls']
    aa = gm.calibration['auxiliary']
    p = gm.calibration['parameters']

    ee = numpy.array([0],dtype=numpy.double)

    N = 1000000

    ss = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ss).T, (1,N)) )
    xx = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(xx).T, (1,N)) )
    aa = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(aa).T, (1,N)) )
    ee = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ee).T, (1,N)) )

    print(ss.shape)

    g = gm.functions['transition']
    f = gm.functions['arbitrage']
    import time

    tmp = g(ss,xx,aa,ee,p)
    t1 = time.time()
    for i in range(50):
        tmp = g(ss,xx,aa,ee,p)
    t2 = time.time()

    print(tmp.shape)
    tmp = f(ss,xx,aa,ss,xx,aa,p)
    t3 = time.time()
    for i in range(50):
        tmp = f(ss,xx,aa,ss,xx,aa,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))

    print(gm)
