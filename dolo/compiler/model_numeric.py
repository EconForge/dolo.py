from dolo.compiler.recipes import recipes


class NumericModel:

    calibration = None
    calibration_dict = None
    covariances = None
    markov_chain = None

    def __init__(self, symbolic_model, options=None, infos=None):

        self.symbolic = symbolic_model

        self.symbols = symbolic_model.symbols

        self.variables = sum( [tuple(e) for k,e in  self.symbols.iteritems() if k not in ('parameters','shocks','values')], ())

        self.options = options if options is not None else {}

        self.infos = infos if infos is not None else {}

        self.infos['data_layout'] = 'columns'

        self.name = self.infos['name']
        self.model_type = self.infos['type']

        self.__update_from_symbolic__()
        self.__compile_functions__()

    def __update_from_symbolic__(self):

        import numpy
        # updates calibration according to the symbolic definitions

        system = self.symbolic.calibration_dict

        from dolo.compiler.triangular_solver import solve_triangular_system
        self.calibration_dict = solve_triangular_system( system )
        from dolo.compiler.misc import calibration_to_vector
        self.calibration = calibration_to_vector(self.symbols, self.calibration_dict)
        from symbolic_eval import NumericEval
        evaluator = NumericEval(self.calibration_dict)

        # read symbolic structure
        covariances = evaluator.eval(self.symbolic.covariances)
        if covariances is None:
            self.covariances = covariances
        else:
            self.covariances = numpy.atleast_2d(numpy.array(covariances, dtype=float))

        markov_chain = evaluator.eval(self.symbolic.markov_chain)
        if markov_chain is None:
            self.markov_chain = None
        else:
            self.markov_chain = [numpy.atleast_2d(numpy.array(tab, dtype=float)) for tab in markov_chain]

        self.options = evaluator.eval(self.symbolic.options)


    def get_calibration(self, pname, *args):

        if isinstance(pname, list):
            return [ self.get_calibration(p) for p in pname ]
        elif isinstance(pname, tuple):
            return tuple( [ self.get_calibration(p) for p in pname ] )
        elif len(args)>0:
            pnames = (pname,) + args
            return self.get_calibration(pnames)

        group = [g for g in self.symbols.keys() if pname in self.symbols[g]]
        try:
            group = group[0]
        except:
            raise Exception('Unknown symbol {}.'.format(pname))
        i = self.symbols[group].index(pname)
        v = self.calibration[group][i]

        return v


    def set_calibration(self, *args, **kwargs):

        # raise exception if unknown symbol ?

        if len(args)==2:
            pname, pvalue = args
            if isinstance(pname, str):
                self.set_calibration(**{pname:pvalue})
        else:
            # else ignore pname and pvalue
            calib =  self.symbolic.calibration_dict
            calib.update(kwargs)
            self.__update_from_symbolic__()

    def __str__(self):

        from dolo.misc.termcolor import colored

        s = u'''
Model object:
------------

- name: "{name}"
- type: "{type}"
- file: "{filename}\n'''.format(**self.infos)

        ss = '\n- residuals:\n\n'
        res = self.residuals()

        for eqgroup, eqlist in self.symbolic.equations.iteritems():
            ss += u"    {}\n".format(eqgroup)
            for i, eq in enumerate(eqlist):
                val = res[eqgroup][i]
                if abs(val) < 1e-8:
                    val = 0

                vals = '{:.4f}'.format(val)

                if abs(val) > 1e-8:
                    vals = colored(vals, 'red')

                # eq = eq.replace('|', u"\u27C2")

                ss += u"        {eqn:3} : {vals} : {eqs}\n".format(eqn=str(i+1), vals=vals, eqs=eq)

            ss += "\n"
        s += ss

        # import pprint
        # s += '- residuals:\n'
        # s += pprint.pformat(compute_residuals(self),indent=2, depth=1)

        return s

    def __repr__(self):
        return self.__str__()

    @property
    def x_bounds(self):

        if 'arbitrage_ub' in self.functions:
            fun_lb = self.functions['arbitrage_lb']
            fun_ub = self.functions['arbitrage_ub']
            return [fun_lb, fun_ub]
        else:
            return None

    def residuals(self, calib=None):

        if self.model_type in ("fg",'fga'):
            from dolo.algos.fg.steady_state import residuals
            return residuals(self, calib)
        elif self.model_type in ('mfg','mfga'):
            from dolo.algos.mfg.steady_state import residuals
            return residuals(self, calib)



    def __compile_functions__(self):

        from dolo.compiler.function_compiler_ast import compile_function_ast
        from dolo.compiler.function_compiler import standard_function

        defs = self.symbolic.definitions

        # works for fg models only
        recipe = recipes[self.model_type]
        symbols = self.symbols # should match self.symbols

        comps = []

        functions = {}

        for funname in recipe['specs'].keys():

            spec = recipe['specs'][funname]

            if funname not in self.symbolic.equations:
                if not spec.get('optional'):
                    raise Exception("The model doesn't contain equations of type '{}'.".format(funname))
                else:
                    continue


            if spec.get('target'):

                # keep only right-hand side
                # TODO: restore recursive definitions
                eqs = self.symbolic.equations[funname]
                eqs = [eq.split('=')[1] for eq in eqs]
                eqs = [str.strip(eq) for eq in eqs]

                target_spec = spec.get('target')
                n_output = len(self.symbols[target_spec[0]])
                # target_short_name = spec.get('target')[2]
                target_spec[2] = 'out'
            else:
                target_spec = None


            if spec.get('complementarities'):

                comp_spec = spec.get('complementarities')
                comp_order = comp_spec['middle']
                comp_args = comp_spec['left-right']

                comps = []
                eqs = []
                for i,eq in enumerate(self.symbolic.equations[funname]):

                    if '|' in eq:
                        control = self.symbols[comp_order[0]][i]
                        eq, comp = str.split(eq,'|')
                        lhs, rhs = decode_complementarity(comp, control)
                        comps.append([lhs, rhs])
                    else:
                        comps.append(['-inf', 'inf'])

                    eqs.append(eq)

                comp_lhs, comp_rhs = zip(*comps)
                fb_names = ['{}_lb'.format(funname), '{}_ub'.format(funname)]

                lower_bound = compile_function_ast(comp_lhs, symbols, comp_args, funname=fb_names[0], use_numexpr=False, definitions=defs)
                upper_bound = compile_function_ast(comp_rhs, symbols, comp_args, funname=fb_names[1], use_numexpr=False, definitions=defs)

                n_output = len(comp_lhs)

                functions[fb_names[0]] = standard_function(lower_bound, n_output )
                functions[fb_names[1]] = standard_function(upper_bound, n_output )


            # rewrite all equations as rhs - lhs
            def filter_equal(eq):
                if '=' in eq:
                    lhs, rhs = str.split(eq,'=')
                    eq = '{} - ( {} )'.format(rhs, lhs)
                    eq = str.strip(eq)
                    return eq
                else:
                    return eq

            eqs = [filter_equal(eq) for eq in eqs]


            arg_names = recipe['specs'][funname]['eqs']

            fun = compile_function_ast(eqs, symbols, arg_names,
                                    output_names=target_spec, funname=funname,
                                        use_numexpr=True, definitions=defs
                                    )

            n_output = len(eqs)

            functions[funname] = standard_function(fun, n_output )

        self.functions = functions


import re
regex = re.compile("(.*)<=(.*)<=(.*)")

def decode_complementarity(comp, control):

    '''
    # comp can be either:
    - None
    - "a<=expr" where a is a controls
    - "expr<=a" where a is a control
    - "expr1<=a<=expr2"
    '''

    try:
        res = regex.match(comp).groups()
    except:
        raise Exception("Unable to parse complementarity condition '{}'".format(comp))
    res = [r.strip() for r in res]
    if res[1] != control:
        msg = "Complementarity condition '{}' incorrect. Expected {} instead of {}.".format(comp, control, res[1])
        raise Exception(msg)
    return [res[0], res[2]]
