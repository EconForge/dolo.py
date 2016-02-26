import ast
from collections import OrderedDict
from .codegen import to_source
from .function_compiler_ast import timeshift, StandardizeDatesSimple
from dolo.compiler.recipes import recipes
from numba import njit

class NumericModel:

    calibration = None
    calibration_dict = None
    covariances = None
    markov_chain = None

    def __init__(self, symbolic_model, options=None, infos=None):

        self.symbolic = symbolic_model

        self.symbols = symbolic_model.symbols

        self.variables = sum( [tuple(e) for k,e in  self.symbols.items() if k not in ('parameters','shocks','values')], ())

        self.options = options if options is not None else {}

        self.infos = infos if infos is not None else {}

        self.infos['data_layout'] = 'columns'

        self.name = self.infos['name']
        self.model_type = self.infos['type']
        # self.model_spec

        self.__update_from_symbolic__()
        self.__compile_functions__()

    def __update_from_symbolic__(self):

        import numpy
        # updates calibration according to the symbolic definitions

        system = self.symbolic.calibration_dict

        from dolo.compiler.triangular_solver import solve_triangular_system
        self.calibration_dict = solve_triangular_system( system )
        from dolo.compiler.misc import CalibrationDict, calibration_to_vector
        calib = calibration_to_vector(self.symbols, self.calibration_dict)
        self.calibration = CalibrationDict(self.symbols, calib)
        from .symbolic_eval import NumericEval
        evaluator = NumericEval(self.calibration_dict)

        # read symbolic structure
        self.options = evaluator.eval(self.symbolic.options)

        distribution = evaluator.eval(self.symbolic.distribution)
        discrete_transition = evaluator.eval(self.symbolic.discrete_transition)


        covariances = distribution
        if distribution is None:
            self.covariances = None
        else:
            self.covariances = numpy.atleast_2d(numpy.array(covariances, dtype=float))

        markov_chain = discrete_transition
        if markov_chain is None:
            self.markov_chain = None
        else:
            self.markov_chain = [numpy.atleast_2d(numpy.array(tab, dtype=float)) for tab in markov_chain]

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

        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            eqlist = self.symbolic.equations[eqgroup]
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

        if 'controls_ub' in self.functions:
            fun_lb = self.functions['controls_lb']
            fun_ub = self.functions['controls_ub']
            return [fun_lb, fun_ub]
        else:
            return None

    def residuals(self, calib=None):

        if self.model_type == 'dtcscc':
            from dolo.algos.dtcscc.steady_state import residuals
            return residuals(self, calib)
        elif self.model_type == 'dtmscc':
            from dolo.algos.dtmscc.steady_state import residuals
            return residuals(self, calib)

    def eval_formula(self, expr, dataframe=None, calib=None):

        from dolo.compiler.eval_formula import eval_formula
        if calib is None:
            calib = self.calibration
        return eval_formula(expr, dataframe=dataframe, context=calib)



    def __compile_functions__(self):

        from dolo.compiler.function_compiler_ast import compile_function_ast
        from dolo.compiler.function_compiler import standard_function

        defs = self.symbolic.definitions

        # works for fg models only
        model_type = self.model_type
        if 'auxiliaries' not in self.symbols:
            model_type += '_'
        else:
            # prepare auxiliaries
            auxeqs = self.symbolic.equations['auxiliary']
            auxdefs = {}
            for time in [-1,0,1]:
                dd = OrderedDict()
                for eq in auxeqs:
                    lhs, rhs = eq.split('=')
                    lhs = ast.parse( str.strip(lhs) ).body[0].value
                    rhs = ast.parse( str.strip(rhs) ).body[0].value
                    tmp = timeshift(rhs, self.variables, time)
                    k = timeshift(lhs, self.variables, time)
                    k = StandardizeDatesSimple(self.variables).visit(k)
                    v = StandardizeDatesSimple(self.variables).visit(tmp)
                    dd[to_source(k)] = to_source(v)
                auxdefs[time] = dd

        recipe = recipes[model_type]
        symbols = self.symbols # should match self.symbols

        comps = []

        functions = {}
        original_functions = {}
        original_gufunctions = {}

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
                if spec.get('recursive') is False:
                    target_spec = None
                else:
                    target_spec[2] = 'out'
            else:
                target_spec = None


            if spec.get('complementarities'):

                # TODO: Rewrite and simplify
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
                # fb_names = ['{}_lb'.format(funname), '{}_ub'.format(funname)]
                fb_names = ['controls_lb'.format(funname), 'controls_ub'.format(funname)]

                ddefs = OrderedDict()
                for ag in comp_args:
                    if ag[0] == 'auxiliaries':
                        t = ag[1]
                        ddefs.update(auxdefs[t])
                ddefs.update(defs)

                lower_bound, gu_lower_bound = compile_function_ast(comp_lhs, symbols, comp_args, funname=fb_names[0],definitions=defs)
                upper_bound, gu_upper_bound = compile_function_ast(comp_rhs, symbols, comp_args, funname=fb_names[1],definitions=defs)

                n_output = len(comp_lhs)

                functions[fb_names[0]] = standard_function(gu_lower_bound, n_output )
                functions[fb_names[1]] = standard_function(gu_upper_bound, n_output )
                original_functions[fb_names[0]] = lower_bound
                original_functions[fb_names[1]] = upper_bound
                original_gufunctions[fb_names[0]] = gu_lower_bound
                original_gufunctions[fb_names[1]] = gu_upper_bound



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


            ddefs = OrderedDict()
            for ag in arg_names:
                if ag[0] == 'auxiliaries':
                    t = ag[1]
                    ddefs.update(auxdefs[t])
            ddefs.update(defs)

            fun, gufun = compile_function_ast(eqs, symbols, arg_names,
                                    output_names=target_spec, funname=funname, definitions=ddefs,
                                    )
            # print("So far so good !")c
            n_output = len(eqs)

            original_functions[funname] = fun

            functions[funname] = standard_function(gufun, n_output )
            original_functions[funname] = fun
            original_gufunctions[funname] = gufun

        self.__original_functions__ = original_functions
        self.__original_gufunctions__ = original_gufunctions
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
