from dolang.symbolic import sanitize
import copy


class SymbolicModel:

    def __init__(self, data):

        self.data = data

    @property
    def symbols(self):
        auxiliaries = [k for k in self.definitions.keys()]
        symbols = {sg: [*self.data['symbols'][sg]] for sg in self.data['symbols'].keys()}
        symbols['auxiliaries'] = auxiliaries
        return symbols

    @property
    def variables(self):
        return sum([
            self.symbols[e] for e in self.symbols.keys() if e != 'parameters'
        ], [])

    @property
    def equations(self):
        vars = self.variables + [*self.definitions.keys()]
        d = dict()
        for g, v in self.data['equations'].items():
            ll = []
            for eq in v:
                if "|" in eq:
                    eq = eq.split("|")[0]
                ll.append(sanitize(eq, variables=vars))
            d[g] = ll

        if "controls_lb" not in d:
            for ind, g in enumerate(("controls_lb", "controls_ub")):
                eqs = []
                for i, eq in enumerate(self.data['equations']['arbitrage']):
                    if "|" not in eq:
                        if ind == 0:
                            eq = "-inf"
                        else:
                            eq = "inf"
                    else:
                        comp = eq.split("|")[1]
                        v = self.symbols["controls"][i]
                        eq = decode_complementarity(comp, v)[ind]
                    eqs.append(eq)
                d[g] = eqs
        return d

    @property
    def definitions(self):
        self.data.get('definitions', {})
        return self.data.get('definitions', {})

    @property
    def name(self):
        return self.data.get("name", "Anonymous")

    @property
    def infos(self):
        infos = {
            'name': self.data.get('name', 'anonymous'),
            'filename': self.data.get('filename', '<string>'),
            'type': 'dtcc'
        }
        return infos

    @property
    def options(self):
        return self.data['options']

    def get_calibration(self):

        symbols = self.symbols
        calibration = self.data.get("calibration", {})
        definitions = self.definitions

        initial_values = {
            'exogenous': 0,
            'expectations': 0,
            'values': 0,
            'controls': float('nan'),
            'states': float('nan')
        }

        # variables defined by a model equation default to using these definitions
        initialized_from_model = {
            'values': 'value',
            'expectations': 'expectation',
            'direct_responses': 'direct_response'
        }

        for k, v in definitions.items():
            if k not in calibration:
                calibration[k] = v

        for symbol_group in symbols:
            if symbol_group not in initialized_from_model.keys():
                if symbol_group in initial_values:
                    default = initial_values[symbol_group]
                else:
                    default = float('nan')
                for s in symbols[symbol_group]:
                    if s not in calibration:
                        calibration[s] = default

        from dolo.compiler.triangular_solver import solve_triangular_system
        return solve_triangular_system(calibration)

    def get_domain(self):

        calibration = self.get_calibration()
        states = self.symbols['states']

        sdomain = self.data.get("domain", {})
        for k in sdomain.keys():
            if k not in states:
                sdomain.pop(k)

        # backward compatibility
        if len(sdomain) == 0 and len(states) > 0:
            try:
                import warnings
                min = get_address(self.data,
                                  ["options:grid:a", "options:grid:min"])
                max = get_address(self.data,
                                  ["options:grid:b", "options:grid:max"])
                for i, s in enumerate(states):
                    sdomain[s] = [min[i], max[i]]
                # shall we raise a warning for deprecated syntax ?
            except Exception as e:
                print(e)
                pass

        if len(sdomain) < len(states):
            missing = [s for s in states if s not in sdomain]
            raise Exception("Missing domain for states: {}.".format(
                str.join(', ', missing)))

        from dolo.compiler.objects import Domain
        from dolo.compiler.language import eval_data
        sdomain = eval_data(sdomain, calibration)
        domain = Domain(**sdomain)
        domain.states = states

        return domain

    def get_exogenous(self):

        if "exogenous" not in self.data:
            return {}

        exo = self.data['exogenous']
        calibration = self.get_calibration()
        from dolo.compiler.language import eval_data
        exogenous = eval_data(exo, calibration)

        from ruamel.yaml.comments import CommentedMap, CommentedSeq
        from dolo.numeric.processes import ProductProcess, Process
        if isinstance(exogenous, Process):
            # old style
            return exogenous
        elif isinstance(exo, list):
            # old style (2)
            return ProductProcess(*exogenous)
        else:
            # new style
            syms = self.symbols['exogenous']
            # first we check that shocks are defined in the right order
            ssyms = []
            for k in exo.keys():
                vars = [v.strip() for v in k.split(',')]
                ssyms.append(vars)
            ssyms = tuple(sum(ssyms, []))
            if tuple(syms)!=ssyms:
                from dolo.compiler.language import ModelError
                lc = exo.lc
                raise ModelError(f"{lc.line}:{lc.col}: 'exogenous' section. Shocks specification must match declaration order. Found {ssyms}. Expected{tuple(syms)}")

            return ProductProcess(*exogenous.values())


    def get_endo_grid(self):

        # determine bounds:
        domain = self.get_domain()
        min = domain.min
        max = domain.max

        options = self.data.get("options", {})

        # determine grid_type
        grid_type = get_type(options.get("grid"))
        if grid_type is None:
            grid_type = get_address(self.data,
                                    ["options:grid:type", "options:grid_type"])
        if grid_type is None:
            raise Exception('Missing grid geometry ("options:grid:type")')

        args = {'min': min, 'max': max}
        if grid_type.lower() in ('cartesian', 'cartesiangrid'):
            from dolo.numeric.grids import UniformCartesianGrid
            orders = get_address(self.data,
                                 ["options:grid:n", "options:grid:orders"])
            if orders is None:
                orders = [20] * len(min)
            grid = UniformCartesianGrid(min=min, max=max, n=orders)
        elif grid_type.lower() in ('nonuniformcartesian', 'nonuniformcartesiangrid'):
            from dolo.compiler.language import eval_data
            from dolo.numeric.grids import NonUniformCartesianGrid
            calibration = self.get_calibration()
            nodes = [eval_data(e, calibration) for e in self.data['options']['grid']]
            print(nodes)
            # each element of nodes should be a vector
            return NonUniformCartesianGrid(nodes)
        elif grid_type.lower() in ('smolyak', 'smolyakgrid'):
            from dolo.numeric.grids import SmolyakGrid
            mu = get_address(self.data, ["options:grid:mu"])
            if mu is None:
                mu = 2
            grid = SmolyakGrid(min=min, max=max, mu=mu)
        else:
            raise Exception("Unknown grid type.")

        return grid


def get_type(d):
    try:
        s = d.tag.value
        return s.strip("!")
    except:
        v = d.get("type")
        return v


def get_address(data, address, default=None):

    if isinstance(address, list):
        found = [get_address(data, e, None) for e in address]
        found = [f for f in found if f is not None]
        if len(found) > 0:
            return found[0]
        else:
            return default
    fields = str.split(address, ':')
    while len(fields) > 0:
        data = data.get(fields[0])
        fields = fields[1:]
        if data is None:
            return default
    return data


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
        raise Exception(
            "Unable to parse complementarity condition '{}'".format(comp))
    res = [r.strip() for r in res]
    if res[1] != control:
        msg = "Complementarity condition '{}' incorrect. Expected {} instead of {}.".format(
            comp, control, res[1])
        raise Exception(msg)
    return [res[0], res[2]]


class Model(SymbolicModel):

    def __init__(self, data, check=True):

        self.data = data
        self.model_type = 'dtcc'
        self.__functions__ = None
        # self.__compile_functions__()
        self.set_changed()
        if check:
            self.calibration
            self.domain
            self.exogenous
            self.x_bounds
            self.functions

    def set_changed(self):
        self.__domain__ = None
        self.__exogenous__ = None
        self.__calibration__ = None

    def set_calibration(self, *pargs, **kwargs):
        if len(pargs)==1:
            self.set_calibration(**pargs[0])
        self.set_changed()
        self.data['calibration'].update(kwargs)

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = super(self.__class__, self).get_calibration()
            from dolo.compiler.misc import CalibrationDict, calibration_to_vector
            calib = calibration_to_vector(self.symbols, calibration_dict)
            self.__calibration__ = CalibrationDict(self.symbols, calib)  #
        return self.__calibration__

    @property
    def exogenous(self):
        if self.__exogenous__ is None:
            self.__exogenous__ = super(self.__class__, self).get_exogenous()
        return self.__exogenous__

    @property
    def domain(self):
        if self.__domain__ is None:
            self.__domain__ = super(self.__class__, self).get_domain()
        return self.__domain__

    def __compile_functions__(self):

        from dolang.function_compiler import make_method_from_factory

        from dolang.vectorize import standard_function
        from dolo.compiler.factories import get_factory

        functions = {}
        original_functions = {}
        original_gufunctions = {}

        funnames = [*self.equations.keys()] + ['auxiliary']

        import dolo.config
        debug = dolo.config.debug

        for funname in funnames:

            fff = get_factory(self, funname)
            fun, gufun = make_method_from_factory(
                fff, vectorize=True, debug=debug)
            n_output = len(fff.content)
            functions[funname] = standard_function(gufun, n_output)
            original_gufunctions[funname] = gufun  # basic gufun function
            original_functions[funname] = fun  # basic numba fun

        self.__original_functions__ = original_functions
        self.__original_gufunctions__ = original_gufunctions
        self.__functions__ = functions

    @property
    def functions(self):
        if self.__functions__ is None:
            self.__compile_functions__()
        return self.__functions__

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

        equations = self.equations.copy()
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])
        definitions = {'definitions': tmp}
        equations.update(definitions)
        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            if eqgroup == 'auxiliary':
                continue
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
                    ss += u" {eqn:2} : {eqs}\n".format(eqn=str(i + 1), eqs=eq)
                else:
                    if abs(val) < 1e-8:
                        val = 0
                    vals = '{:.4f}'.format(val)
                    if abs(val) > 1e-8:
                        vals = colored(vals, 'red')
                    ss += u" {eqn:2} : {vals} : {eqs}\n".format(
                        eqn=str(i + 1), vals=vals, eqs=eq)
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
        </table>""".format(
            name=infos['name'],
            type=infos['type'],
            filename=infos['filename'].replace("<", "&lt").replace(">", "&gt"))

        # Equations and residuals
        resids = self.residuals()
        equations = self.equations.copy()
        # Create definitions equations and append to equations dictionary
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + ' = ' + definitions[deftype])

        definitions = {'definitions': tmp}
        equations.update(definitions)

        variables = sum(
            [e for k, e in self.symbols.items() if k != 'parameters'], [])
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
                        vals = '<span style="color: red;">{:.4f}</span>'.format(
                            val)
                    else:
                        vals = '{:.3f}'.format(val)
                if '|' in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq, '|')
                lat = eq2tex(variables, eq)
                lat = '${}$'.format(lat)
                line = [lat, vals]
                h = eq_type if i == 0 else ''
                fmt_line = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                    h, *line)
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
