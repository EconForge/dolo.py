from dolang.symbolic import sanitize, parse_string, str_expression
from dolang.language import eval_data
from dolang.symbolic import str_expression

import copy


class SymbolicModel:
    def __init__(self, data):

        self.data = data

    @property
    def symbols(self):

        if self.__symbols__ is None:

            from .misc import LoosyDict, equivalent_symbols
            from dolang.symbolic import remove_timing, parse_string, str_expression

            symbols = LoosyDict(equivalences=equivalent_symbols)
            for sg in self.data["symbols"].keys():
                symbols[sg] = [s.value for s in self.data["symbols"][sg]]

            self.__symbols__ = symbols

            # the following call adds auxiliaries (tricky, isn't it?)
            self.definitions

        return self.__symbols__

    @property
    def variables(self):
        if self.__variables__ is None:

            self.__variables__ = sum(
                [self.symbols[e] for e in self.symbols.keys() if e != "parameters"], []
            )

        return self.__variables__

    @property
    def equations(self):
        import yaml.nodes

        if self.__equations__ is None:

            vars = self.variables + [*self.definitions.keys()]

            d = dict()
            for g, v in self.data["equations"].items():

                # new style
                if isinstance(v, yaml.nodes.ScalarNode):
                    assert v.style == "|"
                    if g in ("arbitrage",):
                        start = "complementarity_block"
                    else:
                        start = "assignment_block"
                    eqs = parse_string(v, start=start)
                    eqs = sanitize(eqs, variables=vars)
                    eq_list = eqs.children
                # old style
                else:
                    eq_list = []
                    for eq_string in v:
                        start = "equation"  # it should be assignment
                        eq = parse_string(eq_string, start=start)
                        eq = sanitize(eq, variables=vars)
                        eq_list.append(eq)

                if g in ("arbitrage",):
                    ll = []  # List[str]
                    ll_lb = []  # List[str]
                    ll_ub = []  # List[str]
                    with_complementarity = False
                    for i, eq in enumerate(eq_list):
                        if eq.data == "double_complementarity":
                            v = eq.children[1].children[1].children[0].children[0].value
                            t = int(
                                eq.children[1].children[1].children[1].children[0].value
                            )
                            expected = (
                                self.symbols["controls"][i],
                                0,
                            )  # TODO raise nice error message
                            if (v, t) != expected:
                                raise Exception(
                                    f"Incorrect variable in complementarity: expected {expected}. Found {(v,t)}"
                                )
                            ll_lb.append(str_expression(eq.children[1].children[0]))
                            ll_ub.append(str_expression(eq.children[1].children[2]))
                            eq = eq.children[0]
                            with_complementarity = True
                        else:
                            ll_lb.append("-inf")
                            ll_ub.append("inf")
                        from dolang.symbolic import list_symbols

                        # syms = list_symbols(eq)
                        ll.append(str_expression(eq))
                    d[g] = ll
                    if with_complementarity:
                        d[g + "_lb"] = ll_lb
                        d[g + "_ub"] = ll_ub
                else:
                    # TODO: we should check here that equations are well specified
                    d[g] = [str_expression(e) for e in eq_list]

            # if "controls_lb" not in d:
            #     for ind, g in enumerate(("controls_lb", "controls_ub")):
            #         eqs = []
            #         for i, eq in enumerate(d['arbitrage']):
            #             if "⟂" not in eq:
            #                 if ind == 0:
            #                     eq = "-inf"
            #                 else:
            #                     eq = "inf"
            #             else:
            #                 comp = eq.split("⟂")[1].strip()
            #                 v = self.symbols["controls"][i]
            #                 eq = decode_complementarity(comp, v+"[t]")[ind]
            #             eqs.append(eq)
            #         d[g] = eqs

            self.__equations__ = d

        return self.__equations__

    @property
    def definitions(self):

        from yaml import ScalarNode

        if self.__definitions__ is None:

            # at this stage, basic_symbols doesn't contain auxiliaries
            basic_symbols = self.symbols
            vars = sum(
                [basic_symbols[k] for k in basic_symbols.keys() if k != "parameters"],
                [],
            )

            # # auxiliaries = [remove_timing(parse_string(k)) for k in self.data.get('definitions', {})]
            # # auxiliaries = [str_expression(e) for e in auxiliaries]
            # # symbols['auxiliaries'] = auxiliaries

            if "definitions" not in self.data:
                self.__definitions__ = {}
                # self.__symbols__['auxiliaries'] = []

            elif isinstance(self.data["definitions"], ScalarNode):

                definitions = {}

                # new-style
                from lark import Token

                def_block_tree = parse_string(
                    self.data["definitions"], start="assignment_block"
                )
                def_block_tree = sanitize(
                    def_block_tree
                )  # just to replace (v,) by (v,0) # TODO: remove

                auxiliaries = []
                for eq_tree in def_block_tree.children:
                    lhs, rhs = eq_tree.children
                    tok_name: Token = lhs.children[0].children[0]
                    tok_date: Token = lhs.children[1].children[0]
                    name = tok_name.value
                    date = int(tok_date.value)
                    if name in vars:
                        raise Exception(
                            f"definitions:{tok_name.line}:{tok_name.column}: Auxiliary variable '{name}'' already defined."
                        )
                    if date != 0:
                        raise Exception(
                            f"definitions:{tok_name.line}:{tok_name.column}: Auxiliary variable '{name}' must be defined at date 't'."
                        )
                    # here we could check some stuff
                    from dolang import list_symbols

                    syms = list_symbols(rhs)
                    for p in syms.parameters:
                        if p in vars:
                            raise Exception(
                                f"definitions:{tok_name.line}: Symbol '{p}' is defined as a variable. Can't appear as a parameter."
                            )
                        if p not in self.symbols["parameters"]:
                            raise Exception(
                                f"definitions:{tok_name.line}: Paremeter '{p}' must be defined as a model symbol."
                            )
                    for v in syms.variables:
                        if v[0] not in vars:
                            raise Exception(
                                f"definitions:{tok_name.line}: Variable '{v[0]}[t]' is not defined."
                            )
                    auxiliaries.append(name)
                    vars.append(name)

                    definitions[str_expression(lhs)] = str_expression(rhs)

                self.__symbols__["auxiliaries"] = auxiliaries
                self.__definitions__ = definitions

            else:

                # old style
                from dolang.symbolic import remove_timing

                auxiliaries = [
                    remove_timing(parse_string(k))
                    for k in self.data.get("definitions", {})
                ]
                auxiliaries = [str_expression(e) for e in auxiliaries]
                self.__symbols__["auxiliaries"] = auxiliaries
                vars = self.variables
                auxs = []

                definitions = self.data["definitions"]
                d = dict()
                for i in range(len(definitions.value)):

                    kk = definitions.value[i][0]
                    if self.__compat__:
                        k = parse_string(kk.value)
                        if k.data == "symbol":
                            # TODO: warn that definitions should be timed
                            from dolang.grammar import create_variable

                            k = create_variable(k.children[0].value, 0)
                    else:
                        k = parse_string(kk.value, start="variable")
                    k = sanitize(k, variables=vars)

                    assert k.children[1].children[0].value == "0"

                    vv = definitions.value[i][1]
                    v = parse_string(vv, start="formula")
                    v = sanitize(v, variables=vars)
                    v = str_expression(v)

                    key = str_expression(k)
                    vars.append(key)
                    d[key] = v
                    auxs.append(remove_timing(key))

                self.__symbols__["auxiliaries"] = auxs
                self.__definitions__ = d

        return self.__definitions__

    @property
    def name(self):
        try:
            self.data["name"].value
        except Exception as e:
            return "Anonymous"

    @property
    def infos(self):
        infos = {
            "name": self.name,
            "filename": self.data.get("filename", "<string>"),
            "type": "dtcc",
        }
        return infos

    @property
    def options(self):
        return self.data["options"]

    def get_calibration(self):

        # if self.__calibration__ is None:

        from dolang.symbolic import remove_timing

        import copy

        symbols = self.symbols
        calibration = dict()
        for k, v in self.data.get("calibration", {}).items():
            if v.tag == "tag:yaml.org,2002:str":

                expr = parse_string(v)
                expr = remove_timing(expr)
                expr = str_expression(expr)
            else:
                expr = float(v.value)
            kk = remove_timing(parse_string(k))
            kk = str_expression(kk)

            calibration[kk] = expr

        definitions = self.definitions

        initial_values = {
            "exogenous": 0,
            "expectations": 0,
            "values": 0,
            "controls": float("nan"),
            "states": float("nan"),
        }

        # variables defined by a model equation default to using these definitions
        initialized_from_model = {
            "values": "value",
            "expectations": "expectation",
            "direct_responses": "direct_response",
        }
        for k, v in definitions.items():
            kk = remove_timing(k)
            if kk not in calibration:
                if isinstance(v, str):
                    vv = remove_timing(v)
                else:
                    vv = v
                calibration[kk] = vv

        for symbol_group in symbols:
            if symbol_group not in initialized_from_model.keys():
                if symbol_group in initial_values:
                    default = initial_values[symbol_group]
                else:
                    default = float("nan")
                for s in symbols[symbol_group]:
                    if s not in calibration:
                        calibration[s] = default

        from dolang.triangular_solver import solve_triangular_system

        return solve_triangular_system(calibration)

    #     self.__calibration__ =  solve_triangular_system(calibration)

    # return self.__calibration__

    def get_domain(self):

        calibration = self.get_calibration()
        states = self.symbols["states"]

        sdomain = self.data.get("domain", {})
        for k in sdomain.keys():
            if k not in states:
                sdomain.pop(k)

        # backward compatibility
        if len(sdomain) == 0 and len(states) > 0:
            try:
                import warnings

                min = get_address(self.data, ["options:grid:a", "options:grid:min"])
                max = get_address(self.data, ["options:grid:b", "options:grid:max"])
                for i, s in enumerate(states):
                    sdomain[s] = [min[i], max[i]]
                # shall we raise a warning for deprecated syntax ?
            except Exception as e:
                pass

        if len(sdomain) == 0:
            return None

        if len(sdomain) < len(states):
            missing = [s for s in states if s not in sdomain]
            raise Exception(
                "Missing domain for states: {}.".format(str.join(", ", missing))
            )

        from dolo.compiler.objects import CartesianDomain
        from dolang.language import eval_data

        sdomain = eval_data(sdomain, calibration)

        domain = CartesianDomain(**sdomain)

        return domain

    def get_exogenous(self):

        if "exogenous" not in self.data:
            return {}

        exo = self.data["exogenous"]
        calibration = self.get_calibration()
        from dolang.language import eval_data

        exogenous = eval_data(exo, calibration)

        from dolo.numeric.processes import ProductProcess, Process

        if isinstance(exogenous, Process):
            # old style
            return exogenous
        elif isinstance(exo, list):
            # old style (2)
            return ProductProcess(*exogenous)
        else:
            # new style
            syms = self.symbols["exogenous"]
            # first we check that shocks are defined in the right order
            ssyms = []
            for k in exo.keys():
                vars = [v.strip() for v in k.split(",")]
                ssyms.append(vars)
            ssyms = tuple(sum(ssyms, []))
            if tuple(syms) != ssyms:
                from dolang.language import ModelError

                lc = exo.lc
                raise ModelError(
                    f"{lc.line}:{lc.col}: 'exogenous' section. Shocks specification must match declaration order. Found {ssyms}. Expected{tuple(syms)}"
                )

            return ProductProcess(*exogenous.values())

    @property
    def endo_grid(self):

        # determine bounds:
        domain = self.get_domain()
        min = domain.min
        max = domain.max

        options = self.data.get("options", {})

        # determine grid_type
        grid_type = get_type(options.get("grid"))
        if grid_type is None:
            grid_type = get_address(
                self.data, ["options:grid:type", "options:grid_type"]
            )
        if grid_type is None:
            raise Exception('Missing grid geometry ("options:grid:type")')

        args = {"min": min, "max": max}
        if grid_type.lower() in ("cartesian", "cartesiangrid"):
            from dolo.numeric.grids import UniformCartesianGrid

            orders = get_address(self.data, ["options:grid:n", "options:grid:orders"])
            if orders is None:
                orders = [20] * len(min)
            grid = UniformCartesianGrid(min=min, max=max, n=orders)
        elif grid_type.lower() in ("nonuniformcartesian", "nonuniformcartesiangrid"):
            from dolang.language import eval_data
            from dolo.numeric.grids import NonUniformCartesianGrid

            calibration = self.get_calibration()
            nodes = [eval_data(e, calibration) for e in self.data["options"]["grid"]]
            # each element of nodes should be a vector
            return NonUniformCartesianGrid(nodes)
        elif grid_type.lower() in ("smolyak", "smolyakgrid"):
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
        s = d.tag
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
    fields = str.split(address, ":")
    while len(fields) > 0:
        data = data.get(fields[0])
        fields = fields[1:]
        if data is None:
            return default
    return eval_data(data)


import re

regex = re.compile("(.*)<=(.*)<=(.*)")


def decode_complementarity(comp, control):
    """
    # comp can be either:
    - None
    - "a<=expr" where a is a controls
    - "expr<=a" where a is a control
    - "expr1<=a<=expr2"
    """

    try:
        res = regex.match(comp).groups()
    except:
        raise Exception("Unable to parse complementarity condition '{}'".format(comp))

    res = [r.strip() for r in res]
    if res[1] != control:
        msg = "Complementarity condition '{}' incorrect. Expected {} instead of {}.".format(
            comp, control, res[1]
        )
        raise Exception(msg)

    return [res[0], res[2]]


class Model(SymbolicModel):
    """Model Object"""

    def __init__(self, data, check=True, compat=True):

        self.__compat__ = True

        super().__init__(data)

        self.model_type = "dtcc"
        self.__functions__ = None
        # self.__compile_functions__()
        self.set_changed(all="True")

        if check:
            self.symbols
            self.definitions
            self.calibration
            self.domain
            self.exogenous
            self.x_bounds
            self.functions

    def set_changed(self, all=False):
        self.__domain__ = None
        self.__exogenous__ = None
        self.__calibration__ = None
        if all:
            self.__symbols__ = None
            self.__definitions__ = None
            self.__variables__ = None
            self.__equations__ = None

    def set_calibration(self, *pargs, **kwargs):
        if len(pargs) == 1:
            self.set_calibration(**pargs[0])
        self.set_changed()
        self.data["calibration"].update(kwargs)

    @property
    def calibration(self):
        if self.__calibration__ is None:
            calibration_dict = super().get_calibration()
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
            self.__domain__ = super().get_domain()
        return self.__domain__

    def discretize(self, grid_options=None, dprocess_options={}):

        dprocess = self.exogenous.discretize(**dprocess_options)

        if grid_options is None:
            endo_grid = self.endo_grid
        else:
            endo_grid = self.domain.discretize(**grid_options)

        from dolo.numeric.grids import ProductGrid

        grid = ProductGrid(dprocess.grid, endo_grid, names=["exo", "endo"])
        return [grid, dprocess]

    def __compile_functions__(self):

        from dolang.function_compiler import make_method_from_factory

        from dolang.vectorize import standard_function
        from dolo.compiler.factories import get_factory
        from .misc import LoosyDict

        equivalent_function_names = {
            "equilibrium": "arbitrage",
            "optimality": "arbitrage",
        }
        functions = LoosyDict(equivalences=equivalent_function_names)
        original_functions = {}
        original_gufunctions = {}

        funnames = [*self.equations.keys()]
        if len(self.definitions) > 0:
            funnames = funnames + ["auxiliary"]

        import dolo.config

        debug = dolo.config.debug

        for funname in funnames:

            fff = get_factory(self, funname)
            fun, gufun = make_method_from_factory(fff, vectorize=True, debug=debug)
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

        s = """
        Model:
        ------
        name: "{name}"
        type: "{type}"
        file: "{filename}\n""".format(
            **self.infos
        )

        ss = "\nEquations:\n----------\n\n"
        res = self.residuals()
        res.update({"definitions": zeros(1)})

        equations = self.equations.copy()
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + " = " + definitions[deftype])
        definitions = {"definitions": tmp}
        equations.update(definitions)
        # for eqgroup, eqlist in self.symbolic.equations.items():
        for eqgroup in res.keys():
            if eqgroup == "auxiliary":
                continue
            if eqgroup == "definitions":
                eqlist = equations[eqgroup]
                # Update the residuals section with the right number of empty
                # values. Note: adding 'zeros' was easiest (rather than empty
                # cells), since other variable types have  arrays of zeros.
                res.update({"definitions": [None for i in range(len(eqlist))]})
            else:
                eqlist = equations[eqgroup]
            ss += "{}\n".format(eqgroup)
            for i, eq in enumerate(eqlist):
                val = res[eqgroup][i]
                if val is None:
                    ss += " {eqn:2} : {eqs}\n".format(eqn=str(i + 1), eqs=eq)
                else:
                    if abs(val) < 1e-8:
                        val = 0
                    vals = "{:.4f}".format(val)
                    if abs(val) > 1e-8:
                        vals = colored(vals, "red")
                    ss += " {eqn:2} : {vals} : {eqs}\n".format(
                        eqn=str(i + 1), vals=vals, eqs=eq
                    )
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
            name=infos["name"],
            type=infos["type"],
            filename=infos["filename"].replace("<", "&lt").replace(">", "&gt"),
        )

        # Equations and residuals
        resids = self.residuals()
        equations = self.equations.copy()
        # Create definitions equations and append to equations dictionary
        definitions = self.definitions
        tmp = []
        for deftype in definitions:
            tmp.append(deftype + " = " + definitions[deftype])

        definitions = {"definitions": tmp}
        equations.update(definitions)

        variables = sum([e for k, e in self.symbols.items() if k != "parameters"], [])
        table = '<tr><td><b>Type</b></td><td style="width:80%"><b>Equation</b></td><td><b>Residual</b></td></tr>\n'

        for eq_type in equations:

            eq_lines = []
            for i in range(len(equations[eq_type])):
                eq = equations[eq_type][i]
                # if eq_type in ('expectation','direct_response'):
                #     vals = ''
                if eq_type not in ("arbitrage", "transition", "arbitrage_exp"):
                    vals = ""
                else:
                    val = resids[eq_type][i]
                    if abs(val) > 1e-8:
                        vals = '<span style="color: red;">{:.4f}</span>'.format(val)
                    else:
                        vals = "{:.3f}".format(val)
                if "⟂" in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq, "⟂")
                if "|" in eq:
                    # keep only lhs for now
                    eq, comp = str.split(eq, "|")
                lat = eq2tex(variables, eq)
                lat = "${}$".format(lat)
                line = [lat, vals]
                h = eq_type if i == 0 else ""
                fmt_line = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(h, *line)
                #         print(fmt_line)
                eq_lines.append(fmt_line)
            table += str.join("\n", eq_lines)
        table = "<table>{}</table>".format(table)

        return table_infos + table

    @property
    def x_bounds(self):

        if "controls_ub" in self.functions:
            fun_lb = self.functions["controls_lb"]
            fun_ub = self.functions["controls_ub"]
            return [fun_lb, fun_ub]
        elif "arbitrage_ub" in self.functions:
            fun_lb = self.functions["arbitrage_lb"]
            fun_ub = self.functions["arbitrage_ub"]
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
