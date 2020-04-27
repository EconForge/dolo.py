from dolo.compiler.recipes import recipes
import numpy as np

class Model:

    calibration = None
    calibration_dict = None

    def __init__(self, symbolic_model, options=None, infos=None):

        self.symbolic = symbolic_model
        self.symbols = symbolic_model.symbols

        # compat
        if self.symbolic.definitions:
            self.symbols['auxiliaries'] = [e for e in self.symbolic.definitions.keys()]

        self.variables = sum( [tuple(e) for k,e in  self.symbols.items() if k not in ('parameters','shocks','exogenous', 'values')], ())

        self.options = options if options is not None else {}

        self.infos = infos
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
        self.exogenous = self.get_exogenous()
        self.domain = self.get_domain()



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
        except Exception:
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
        from numpy import zeros
        s = u'''
Model:
------
name: "{name}"
type: "{type}"
file: "{filename}\n'''.format(**self.infos)

        ss = '\nEquations:\n----------\n\n'
        res = self.residuals()
        res.update({'definitions': zeros(1)})

        if self.model_type == 'dynare':
            equations = {"dynare": self.symbolic.equations.copy()}
        else:
            equations = self.symbolic.equations.copy()
        definitions = self.symbolic.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])
        definitions = {'definitions': tmp}
        equations.update(definitions)
        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            if eqgroup == 'auxiliary':
                continue
            if eqgroup == 'dynare':
                eqlist = equations
            if eqgroup == 'definitions':
                eqlist = equations[eqgroup]
                # Update the residuals section with the right number of empty
                # values. Note: adding 'zeros' was easiest (rather than empty
                # cells), since other variable types have  arrays of zeros.
                res.update({'definitions': [None for i in range(len(eqlist))]})
            else:
                eqlist = equations[eqgroup]
            ss += u"{}\n".format(eqgroup)
            for i, eq in enumerate(eqlist):
                val = res[eqgroup][i]
                if val is None:
                    ss += u" {eqn:2} : {eqs}\n".format(eqn=str(i+1), eqs=eq)
                else:
                    if abs(val) < 1e-8:
                        val = 0
                    vals = '{:.4f}'.format(val)
                    if abs(val) > 1e-8:
                        vals = colored(vals, 'red')
                    ss += u" {eqn:2} : {vals} : {eqs}\n".format(eqn=str(i+1), vals=vals, eqs=eq)
            ss += "\n"
        s += ss

        return s

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):

        from dolang.latex import eq2tex

        # general informations
        infos = self.infos
        table_infos = """
     <table>
         <td><b>Model</b></td>
     <tr>
        <td>name</td>
        <td>{name}</td>
      </tr>
    <tr>
        <td>type</td>
        <td>{type}</td>
      </tr>
      <tr>
        <td>filename</td>
        <td>{filename}</td>
      </tr>
    </table>""".format(name=infos['name'], type=infos['type'], filename=infos['filename'].replace("<","&lt").replace(">","&gt"))


        # Equations and residuals
        resids = self.residuals()
        if self.model_type == 'dynare':
            equations = {"dynare": self.symbolic.equations.copy()}
        else:
            equations = self.symbolic.equations.copy()
        # Create definitions equations and append to equations dictionary
        definitions = self.symbolic.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])

        definitions = {'definitions': tmp}
        equations.update(definitions)

        variables = sum([e for k,e in self.symbols.items() if k != 'parameters'], [])
        table = "<tr><td><b>Type</b></td><td><b>Equation</b></td><td><b>Residual</b></td></tr>\n"

        for eq_type in equations:

            eq_lines = []
            for i in range(len(equations[eq_type])):
                eq = equations[eq_type][i]
                # if eq_type in ('expectation','direct_response'):
                #     vals = ''
                if eq_type not in ('arbitrage', 'transition', 'arbitrage_exp'):
                    vals = ''
                else:
                    val = resids[eq_type][i]
                    if abs(val) > 1e-8:
                        vals = '<span style="color: red;">{:.4f}</span>'.format(val)
                    else:
                        vals = '{:.3f}'.format(val)
                if '|' in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq,'|')
                lat = eq2tex(variables, eq)
                lat = '${}$'.format(lat)
                line = [lat, vals]
                h = eq_type if i==0 else ''
                fmt_line = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(h, *line)
        #         print(fmt_line)
                eq_lines.append(fmt_line)
            table += str.join("\n", eq_lines)
        table = "<table>{}</table>".format(table)

        return table_infos + table

    @property
    def x_bounds(self):

        if 'controls_ub' in self.functions:
            fun_lb = self.functions['controls_lb']
            fun_ub = self.functions['controls_ub']
            return [fun_lb, fun_ub]
        else:
            return None

    def residuals(self, calib=None):

        from dolo.algos.steady_state import residuals
        return residuals(self, calib)

    def eval_formula(self, expr, dataframe=None, calib=None):

        from dolo.compiler.eval_formula import eval_formula
        if calib is None:
            calib = self.calibration
        return eval_formula(expr, dataframe=dataframe, context=calib)

    def get_exogenous(model, **opts):
        from dolo.compiler.objects import IIDProcess
        import copy
        gg = model.symbolic.exogenous
        if gg is None:
            raise Exception("Model has no exggenous process.")
        d = copy.deepcopy(gg)
        d.update(opts)
        if 'type' in d: d.pop('type')
        obj =  d.eval(model.calibration.flat)
        if not isinstance(obj, IIDProcess):
            raise Exception("Exgogenous shocks don't follow an IID process.")
        else:
            return obj

    def get_domain(model):

        sdomain = model.symbolic.domain
        states = model.symbols['states']
        from dolo.compiler.language import Domain
        d = Domain(**sdomain)
        domain = d.eval(d=model.calibration.flat)
        # a bit of a hack...
        for k in domain.keys():
            if k not in states:
                domain.pop(k)
        return domain

    def get_endo_grid(model, **dis_opts):
        import copy
        domain = model.get_domain()
        a = np.array([e[0] for e in domain.values()])
        b = np.array([e[1] for e in domain.values()])

        gg = model.symbolic.options.get('grid',{})

        d = copy.deepcopy(gg)
        gtype = dis_opts.get('type')
        if gtype:
            from dolo.compiler.language import minilang
            try:
                cls = [e for e in minilang if e.__name__.lower()==gtype.lower()][0]
            except:
                raise Exception("Unknown grid type {}.".format(gtype))
            d = cls(**d)
    #     return cc
        d.update(**dis_opts)
        if 'a' not in d.keys():
            d['min'] = a
        if 'b' not in d.keys():
            d['max'] = b
        if 'type' in d: d.pop('type')
        grid = d.eval(d=model.calibration.flat)
        # temporary
        from dolo.numeric.grids import CartesianGrid, SmolyakGrid
        if 'Cartesian' in str(grid.__class__):
            return CartesianGrid(grid.a, grid.b, grid.orders)
        if 'Smolyak' in str(grid.__class__):
            return SmolyakGrid(grid.a, grid.b, grid.mu)

    def __compile_functions__(self):

        from dolang.function_compiler import compile_function_ast, standard_function

        defs = self.symbolic.definitions

        model_type = self.model_type

        recipe = recipes[model_type]
        symbols = self.symbols # should match self.symbols

        comps = []

        functions = {}
        original_functions = {}
        original_gufunctions = {}

        # for funname in recipe['specs'].keys():
        for funname in self.symbolic.equations:

            spec = recipe['specs'][funname]

            eqs = self.symbolic.equations[funname]

            if spec.get('target'):
                target = spec['target']
                rhs_only = True
            else:
                target = None
                rhs_only = False

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

                    lower_bound, gu_lower_bound = compile_function_ast(comp_lhs, symbols, comp_args, funname=fb_names[0],definitions=defs)
                    upper_bound, gu_upper_bound = compile_function_ast(comp_rhs, symbols, comp_args, funname=fb_names[1],definitions=defs)

                    n_output = len(comp_lhs)

                    functions[fb_names[0]] = standard_function(gu_lower_bound, n_output )
                    functions[fb_names[1]] = standard_function(gu_upper_bound, n_output )
                    original_functions[fb_names[0]] = lower_bound
                    original_functions[fb_names[1]] = upper_bound
                    original_gufunctions[fb_names[0]] = gu_lower_bound
                    original_gufunctions[fb_names[1]] = gu_upper_bound


            arg_names = recipe['specs'][funname]['eqs']

            fun, gufun = compile_function_ast(eqs, symbols, arg_names, output_names=target, rhs_only=rhs_only, funname=funname, definitions=defs,                                   )
            n_output = len(eqs)
            original_functions[funname] = fun
            functions[funname] = standard_function(gufun, n_output )
            original_functions[funname] = fun
            original_gufunctions[funname] = gufun

        # temporary hack to keep auxiliaries
        auxiliaries = [k for k in defs.keys()]
        symbols['auxiliaries'] = auxiliaries
        eqs = ['{} = {}'.format(k,k) for k in auxiliaries]
        if model_type == 'dtcscc':
            arg_names = [('states',0,'s'),('controls',0,'x'),('parameters',0,'p')]
        elif model_type=='dtcc':
            arg_names = [('exogenous',0,'m'),('states',0,'s'),('controls',0,'x'),('parameters',0,'p')]
        target = ('auxiliaries',0,'y')
        rhs_only = True
        funname = 'auxiliary'
        fun, gufun = compile_function_ast(eqs, symbols, arg_names, output_names=target,
            rhs_only=rhs_only, funname=funname, definitions=defs)
        n_output = len(eqs)
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

# compat
NumericModel = Model
