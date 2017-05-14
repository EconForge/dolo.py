import dolo.config
dolo.config.debug = True
import ruamel.yaml as ry
from ruamel.yaml.comments import CommentedSeq, CommentedMap

with open("examples/models/rbc00.yaml") as f:
    txt = f.read()

print( txt )

txt = txt.replace("^","**")

data = ry.load(txt, ry.RoundTripLoader)
data['filename'] = "examples/models/rbc00.yaml"

data.__class__


import copy

class SymbolicModel:

    def __init__(self, data):

        self.data = data

    @property
    def symbols(self):
        auxiliaries = [k for k in self.definitions.keys()]
        symbols = copy.deepcopy( self.data['symbols'] )
        symbols['auxiliaries'] = auxiliaries
        return symbols

    @property
    def equations(self):
        return self.data['equations']

    @property
    def definitions(self):
        return self.data.get('definitions', {})

    @property
    def name(self):
        return self.data.get("name", "Anonymous")

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

        calibration = self.data.get("calibration", {})
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
        if len(sdomain)==0 and len(states)>0:
            try:
                import warnings
                min = get_address(self.data, ["options:grid:a", "options:grid:min"])
                max = get_address(self.data, ["options:grid:b", "options:grid:max"])
                for i,s in enumerate(states):
                    sdomain[s] = [min[i], max[i]]
                # shall we raise a warning for deprecated syntax ?
            except Exception as e:
                print(e)
                pass

        if len(sdomain)<len(states):
            missing = [s for s in states if s not in sdomain]
            raise Exception("Missing domain for states: {}.".format(str.join(', ', missing)))

        from dolo.compiler.language import Domain

        d = Domain(**sdomain)
        domain = d.eval(d=calibration)
        domain.states = states

        return domain


    def get_exogenous(self):

        exo = self.data.get("exogenous", {})
        calibration = self.get_calibration()
        type = get_type(exo)
        from dolo.compiler.language import Normal, AR1, MarkovChain
        if type == "Normal":
            exog = Normal(**exo)
        elif type in ("AR1", "VAR1"):
            exog = AR1(**exo)
        elif type == "MarkovChain":
            exog = MarkovChain(**exo)
        d = exog.eval(d=calibration)
        return d


    def get_grid(self):

        states = self.symbols['states']

        # determine bounds:
        domain = self.get_domain()
        min = domain.min
        max = domain.max

        options = self.data.get("options", {})

        # determine grid_type
        grid_type = get_type(options.get("grid"))
        if grid_type is None:
            grid_type = get_address(self.data, ["options:grid:type", "options:grid_type"])
        if grid_type is None:
            raise Exception('Missing grid geometry ("options:grid:type")')

        args = {'min': min, 'max': max}
        if grid_type.lower() in ('cartesian', 'cartesiangrid'):
            # from dolo.numerid.grids import CartesianGrid
            from dolo.numeric.grids import CartesianGrid
            orders = get_address(self.data, ["options:grid:n", "options:grid:orders"])
            if orders is None:
                orders = [20]*len(min)
            grid = CartesianGrid(min=min, max=max, n=orders)
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
        if len(found)>0:
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

#
# smodel = SymbolicModel(data)
# smodel.data.get('symbols')
# smodel.get_calibration()
# smodel.get_domain()
# smodel.get_exogenous()
# grid = smodel.get_grid()
#
# print(grid)


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



class Model(SymbolicModel):

    def __init__(self, data):

        self.data = data
        self.model_type = 'dtcc'
        self.__compile_functions__()
        self.set_changed()

    def set_changed(self):
        self.__domain__ = None
        self.__exogenous__ = None
        self.__calibration__ = None

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = super(self.__class__, self).get_calibration()
            from dolo.compiler.misc import CalibrationDict, calibration_to_vector
            calib = calibration_to_vector(self.symbols, calibration_dict)
            self.__calibration__ = CalibrationDict(self.symbols, calib)        #
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

        from dolang.function_compiler import compile_function_ast, standard_function
        from dolo.compiler.recipes import recipes

        defs = self.definitions

        model_type = self.model_type

        recipe = recipes[model_type]
        symbols = self.symbols # should match self.symbols

        comps = []

        functions = {}
        original_functions = {}
        original_gufunctions = {}

        # for funname in recipe['specs'].keys():
        for funname in self.equations:

            spec = recipe['specs'][funname]

            eqs = self.equations[funname]

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
                for i,eq in enumerate(self.equations[funname]):

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
        #
        # temporary hack to keep auxiliaries
        auxiliaries = symbols['auxiliaries']
        eqs = ['{} = {}'.format(k,k) for k in auxiliaries]
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
                equations = self.equations.copy()
                # Create definitions equations and append to equations dictionary
                definitions = self.definitions
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

                # # compat
                # if len(self.definitions)>0:
                #     self.symbols['auxiliaries'] = [e for e in self.symbolic.definitions.keys()]
                # self.variables = sum( [tuple(e) for k,e in  self.symbols.items() if k not in ("parameters", "shocks", "exogenous")], ())
                    #

                    #
                    # self.options = options if options is not None else {}
                    #
                    # self.infos = infos
                    # self.name = self.infos['name']
                    # self.model_type = self.infos['type']
                    # # self.model_spec
                    # self.__update_from_symbolic__()
                    # self.__compile_functions__()


model = Model(data)
print(model.definitions)
print(model.name)
print(model.model_type)
print("Calibration")
print(model.calibration['states'])
print(model.domain)

m,s,x,p = model.calibration['exogenous','states','controls','parameters']


S = model.functions['transition'](m,s,x,m,p)

print( S )

y = model.functions['auxiliary'](m,s,x,p)
print(y)

print(model)

from dolo import time_iteration
time_iteration(model, verbose=True)
print(model._repr_html_())
