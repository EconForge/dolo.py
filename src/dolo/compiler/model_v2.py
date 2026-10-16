"""
``ModelV2`` — the dolo model object rebuilt on ``dyno.dynspec`` + JAX.

Migration plan status (design decisions D1-D4, see planning discussion):

* equations are parsed, checked and ordered by ``dyno.dynspec`` (recipe
  conformity checking, definitions DAG);
* evaluation and derivatives are JAX-based (interpreted evaluation first —
  see ``dolo.compiler.functions_v2``);
* the public API (``symbols``/``equations``/``definitions``/``calibration``/
  ``functions``/``exogenous``/``domain``/``discretize``/``residuals`` ...
  and the legacy function signatures) mirrors
  :class:`dolo.compiler.model.Model` so that ``dolo.algos`` can run
  unchanged;
* TEMPORARY: dolo-YAML specific equation syntax (``⟂`` bounds, ``==``,
  untimed symbols, ``x(1)``) is normalized by
  ``dolo.compiler.equation_preprocess`` before dynspec parsing, until the
  dynspec grammar grows native support;
* TEMPORARY: tagged objects (``!Normal``, ``!Cartesian``, ...) still come
  from the dolang ``language_element`` machinery via
  ``dolo.compiler.langspec`` (imported through ``dolo.compiler.objects``,
  ``dolo.numeric.processes`` and ``dolo.numeric.distribution``).

Two entry points are supported:

``ModelV2.from_yaml(data, filename)``
    A dolo YAML model loaded with ``yaml.safe_load`` (sections ``symbols``,
    ``equations``, ``definitions``, ``calibration``, ``exogenous``,
    ``domain``, ``options``).

``ModelV2.from_dyno(text, filename)``
    A ``.dyno`` file parsed by ``dyno``: parameters are top-level
    assignments, steady states use the ``n[~]`` syntax, shocks are assigned
    through ``e[t] <- N(...)``, and equation types are denoted by block
    tags (``[definitions] :: {...}``, ``[transition] :: {...}``,
    ``[arbitrage] :: {...}``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from lark.visitors import Interpreter

from dyno.dynspec.grammar import parser, str_expression
from dyno.dynspec.analyze import FormulaEvaluator
from dyno.dynspec.recipe import extract_lhs_variable

from dolo.compiler import equation_preprocess as _pre
from dolo.compiler import functions_v2
from dolo.compiler.langspec import eval_data
from dolo.compiler.misc import CalibrationDict, LoosyDict
from dolo.compiler.objects import CartesianDomain

# Tagged objects (`!Normal`, `!Product`, ...) are registered in dolang's
# language registry when these modules are imported.
import dolo.compiler.objects  # noqa: F401
from dolo.numeric import processes, distribution  # noqa: F401

DEFAULT_INITIAL_VALUES = {
    "exogenous": 0.0,
    "expectations": 0.0,
    "values": 0.0,
    "controls": np.nan,
    "states": np.nan,
}


class TaggedDict(dict):
    """A plain dict that remembers its YAML tag (e.g. ``!Cartesian``)."""

    def __init__(self, tag: str, mapping: dict) -> None:
        super().__init__(mapping)
        self.tag = tag


def yaml_compose(txt: str):
    """Compose a dolo YAML model into a node tree (dolang-style)."""
    return yaml.compose(txt)


def _scalar_value(scalar_node) -> Any:
    if isinstance(scalar_node, str):
        return scalar_node
    if scalar_node.tag in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float"):
        return float(scalar_node.value)
    return scalar_node.value


def _key_text(k) -> str:
    return k if isinstance(k, str) else k.value


def _node_to_native(node) -> Any:
    """Convert a (mostly plain) YAML node tree into python data.

    Tagged constructs are either resolved as language elements (through
    ``langspec.eval_data``) or kept as :class:`TaggedDict`.
    """
    from yaml import MappingNode, SequenceNode, ScalarNode

    if isinstance(node, ScalarNode):
        return _scalar_value(node)
    if isinstance(node, SequenceNode):
        if node.tag != "tag:yaml.org,2002:seq":
            return eval_data(node, {})
        return [_node_to_native(c) for c in node]
    if isinstance(node, MappingNode):
        if node.tag != "tag:yaml.org,2002:map":
            try:
                return eval_data(node, {})
            except Exception:
                inner = {_key_text(k): _node_to_native(v) for k, v in node.items()}
                return TaggedDict(node.tag, inner)
        return {_key_text(k): _node_to_native(v) for k, v in node.items()}
    raise Exception(f"Unknown yaml node: {node}")


# ---------------------------------------------------------------------------
# symbol collection over dynspec trees
# ---------------------------------------------------------------------------


class _SymbolCollector(Interpreter):
    """Collect constant and variable names appearing in a dynspec tree."""

    def __init__(self) -> None:
        super().__init__()
        self.names: set[str] = set()

    @staticmethod
    def _name_of(name_node) -> str:
        # `cname` is a kept rule (`!cname`), so symbol nodes wrap a
        # `name` tree which wraps the actual `NAME` token.
        return str(name_node.children[0])

    def constant(self, tree) -> None:
        self.names.add(self._name_of(tree.children[0]))

    def variable(self, tree) -> None:
        self.names.add(self._name_of(tree.children[0]))

    def value(self, tree) -> None:
        self.names.add(self._name_of(tree.children[0]))


def list_symbols(tree) -> set[str]:
    collector = _SymbolCollector()
    collector.visit(tree)
    return collector.names


class _TimedVariableCollector(Interpreter):
    """Collect (name, shift) pairs of timed variables in a dynspec tree."""

    def __init__(self) -> None:
        super().__init__()
        self.pairs: set[tuple[str, int]] = set()

    def variable(self, tree) -> None:
        name = _SymbolCollector._name_of(tree.children[0])
        timing = tree.children[2].children
        shift = int(str(timing[0])) if timing else 0
        self.pairs.add((name, shift))


def _timed_variables(tree) -> set[tuple[str, int]]:
    collector = _TimedVariableCollector()
    collector.visit(tree)
    return collector.pairs


# ---------------------------------------------------------------------------
# scalar evaluation / calibration solving
# ---------------------------------------------------------------------------


def _eval_scalar(expr: str, known: Dict[str, float]) -> float:
    """Evaluate an untimed formula (all names are constants)."""
    expr = str(expr).strip()
    if expr == "inf":
        return np.inf
    tree = parser.parse(expr, start="formula")
    ev = FormulaEvaluator(
        {"constants": dict(known), "variables": {}, "values": {}, "processes": {}}
    )
    return float(ev.visit(tree))


def _solve_symbolic_dict(
    calibration: Dict[str, Any],
    definitions: Dict[str, str],
    symbols: Dict[str, List[str]],
) -> Dict[str, float]:
    """Resolve a mixed value/expression calibration dict iteratively.

    Every name either holds a numeric value or an expression over other
    names; expressions are evaluated as soon as every name they reference
    is known.  This replaces
    ``dolang.triangular_solver.solve_triangular_system``.
    """
    known: Dict[str, float] = {}
    pending: Dict[str, str] = {}

    for name, value in calibration.items():
        if isinstance(value, str) and not _is_numeric(value):
            pending[str(name)] = str(value)
        else:
            known[str(name)] = float(value)

    # definitions also provide expressions for their auxiliary variables
    for lhs, rhs in definitions.items():
        if lhs not in known and lhs not in pending:
            pending[lhs] = rhs

    while pending:
        progressed = False
        for name in list(pending.keys()):
            tree = parser.parse(pending[name].strip(), start="formula")
            needed = list_symbols(tree) - set(known)
            if needed:
                continue
            known[name] = _eval_scalar(pending[name], known)
            del pending[name]
            progressed = True
        if not progressed:
            missing = ", ".join(sorted(pending.keys()))
            raise Exception(
                "Calibration equations cannot be solved (circular or "
                f"undefined references): {missing}"
            )
    return known


def _is_numeric(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _calibration_grouped(
    symbols: Dict[str, List[str]], solved: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Assemble solved values into per-group vectors (dolo convention)."""
    grouped: Dict[str, np.ndarray] = {}
    for group, names in symbols.items():
        grouped[group] = np.array([solved.get(name, np.nan) for name in names])
    return grouped


# ---------------------------------------------------------------------------
# the model object
# ---------------------------------------------------------------------------


class ModelV2:
    """Dolo model built on ``dyno.dynspec`` (see module docstring)."""

    model_type = "dtcc"

    def __init__(
        self,
        *,
        symbols: Dict[str, List[str]],
        equations: Dict[str, List[str]],
        definitions: Dict[str, str],
        calibration: Dict[str, Any],
        exogenous: Any = None,
        domain: Any = None,
        options: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        filename: str = "<string>",
        check: bool = True,
        bounds: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        self.symbols = {k: list(v) for k, v in symbols.items()}
        self.__equations_text__ = {k: list(v) for k, v in equations.items()}
        self.__definitions_text__ = dict(definitions)
        self.__calibration_data__ = dict(calibration)
        self.__exogenous_object__ = exogenous
        self.__domain__ = domain
        self.__options_node__ = options
        self.__options__ = None
        self.name = name if name is not None else "Anonymous"
        self.filename = filename

        self.__equations_trees__: Optional[Dict[str, List[Any]]] = None
        self.__bound_text__: Dict[str, List[str]] = {}
        self.__bound_trees__: Dict[str, List[Any]] = {}
        # parsed complementarity bounds (from `.dyno` import): group name
        # (`arbitrage_lb`/`arbitrage_ub`/…) -> bound trees, one per equation.
        if bounds:
            for bname, trees in bounds.items():
                self.__bound_trees__[bname] = list(trees)
                self.__bound_text__[bname] = [str_expression(t) for t in trees]
        self.__calibration__ = None
        self.__exogenous__ = None
        self.__functions__ = None
        self.__defs_block__: Optional[functions_v2.DefinitionsBlock] = None

        if check:
            self.equations
            self.definitions
            self.calibration
            self.exogenous
            self.domain
            self.functions
            self.x_bounds

    # ------------------------------------------------------------------
    # entry points
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        data: Dict[str, Any] | Any,
        filename: str = "<string>",
        check: bool = True,
    ) -> "ModelV2":
        # `data` is the node tree produced by `yaml.compose` (the same tree
        # the legacy importer consumes); converting each section keeps the
        # tagged-object semantics of dolang's language elements.
        from yaml import MappingNode, ScalarNode, SequenceNode

        symbols = {
            _key_text(key): [s.value for s in data["symbols"][key]]
            for key in data["symbols"].keys()
        }
        definitions = _normalize_definitions_nodes(data.get("definitions"))
        if definitions:
            symbols.setdefault("auxiliaries", list(definitions.keys()))

        equations: Dict[str, List[str]] = {}
        eq_section = data.get("equations")
        for key in eq_section.keys() if eq_section is not None else []:
            block = eq_section[key]
            if isinstance(block, ScalarNode):
                equations[_key_text(key)] = _pre.split_equations(block.value)
            elif isinstance(block, SequenceNode):
                equations[_key_text(key)] = _pre.split_equations(
                    [_scalar_value(c) for c in block]
                )
            else:
                raise Exception(f"Unexpected `equations` entry: {block}")

        calibration: Dict[str, Any] = {}
        cal_section = data.get("calibration")
        for k, v in cal_section.items() if cal_section is not None else []:
            calibration[_key_text(k)] = _scalar_value(v)

        exogenous_node = data.get("exogenous")
        domain_node = data.get("domain")
        options_node = data.get("options")

        return cls(
            symbols=symbols,
            equations=equations,
            definitions=definitions,
            calibration=calibration,
            exogenous=exogenous_node,
            domain=domain_node,
            options=options_node,
            name=_name_of(data.get("name")),
            filename=filename,
            check=check,
        )

    @classmethod
    def from_dyno(
        cls, text: str, filename: str = "<string>", check: bool = True
    ) -> "ModelV2":
        from dyno.larkfiles import DynoFile

        dynofile = DynoFile(text, filename)
        context = dynofile.context

        # --- classify equations by their block tags --------------------
        # Equation types are denoted by tags: [definitions]/[transition]/
        # [arbitrage]/[value]/...
        grouped: Dict[str, List[Any]] = {
            name: [] for name in list(functions_v2.GROUP_LAYOUTS) + ["definitions"]
        }
        for eq in dynofile.equations:
            meta = getattr(eq.meta, "statement_metadata", {}) or {}
            tags = set(meta.get("tags", []))
            for group in grouped:
                if group in tags:
                    grouped[group].append(eq)
                    break
            # untagged/unknown-tagged equations are not dolo equation groups
            # and are ignored

        # --- infer symbol groups ---------------------------------------
        parameters = list(context["constants"].keys())
        exogenous_names: List[str] = []
        for names in context["processes"].keys():
            for name in names:
                if name not in exogenous_names:
                    exogenous_names.append(name)
        states = [
            lhs
            for eq in grouped["transition"]
            if (lhs := extract_lhs_variable(eq)) is not None
        ]
        auxiliaries = [
            lhs
            for eq in grouped["definitions"]
            if (lhs := extract_lhs_variable(eq)) is not None
        ]
        reserved = (
            set(parameters) | set(exogenous_names) | set(states) | set(auxiliaries)
        )
        controls: List[str] = []
        # controls = free variables appearing in any equation group, in the
        # order they show up in the arbitrage block first, then the rest.
        scan_order = (
            ["arbitrage"]
            + [g for g in grouped if g not in ("arbitrage", "definitions")]
            + ["definitions"]
        )
        for group in scan_order:
            for eq in grouped[group]:
                for name in extract_variables_in(eq):
                    if name not in reserved and name not in controls:
                        controls.append(name)

        symbols = {
            "exogenous": exogenous_names,
            "states": states,
            "controls": controls,
            "parameters": parameters,
        }
        if auxiliaries:
            symbols["auxiliaries"] = auxiliaries

        equations_text: Dict[str, List[str]] = {}
        bounds: Dict[str, List[Any]] = {}
        for group, eqs in grouped.items():
            if group == "definitions" or len(eqs) == 0:
                continue
            texts: List[str] = []
            lbs: List[Any] = []
            ubs: List[Any] = []
            any_bound = False
            for i, eq in enumerate(eqs):
                comp = _complementarity_parts(eq)
                if comp is None:
                    texts.append(_tree_text(eq))
                    lbs.append(None)
                    ubs.append(None)
                    continue
                residual, lb, xname, ub = comp
                if group == "arbitrage" and controls and i < len(controls):
                    if xname != controls[i]:
                        raise Exception(
                            f"Complementarity condition in `[arbitrage]` "
                            f"equation {i + 1} is incorrect. Expected "
                            f"'{controls[i]}' instead of '{xname}'."
                        )
                texts.append(_tree_text(residual))
                lbs.append(lb)
                ubs.append(ub)
                any_bound = True
            equations_text[group] = texts
            if any_bound:
                bounds[f"{group}_lb"] = [
                    t if t is not None else _neg_inf_tree() for t in lbs
                ]
                bounds[f"{group}_ub"] = [
                    t if t is not None else _inf_tree() for t in ubs
                ]

        definitions: Dict[str, str] = {}
        for eq in grouped["definitions"]:
            lhs = extract_lhs_variable(eq)
            if lhs is not None:
                definitions[lhs] = _equation_rhs_text(eq)

        # --- calibration from constants + steady states ----------------
        calibration = dict(context["constants"])
        steady = context.get("steady_states", {})
        for group, names in symbols.items():
            if group == "parameters":
                continue
            default = DEFAULT_INITIAL_VALUES.get(group, np.nan)
            for name in names:
                calibration[name] = steady.get(name, default)

        exogenous = _convert_processes(context.get("processes", {}))

        # @options: / @domain: one-liners are exposed as plain dicts
        # through dyno's metadata machinery.
        options = dynofile.metadata.get("options") or {}
        domain = dynofile.metadata.get("domain")

        model = cls(
            symbols=symbols,
            equations=equations_text,
            definitions=definitions,
            calibration=calibration,
            exogenous=exogenous,
            domain=domain,
            options=options,
            name=dynofile.metadata.get("name"),
            filename=filename,
            check=check,
            bounds=bounds,
        )
        model.__dyno_context__ = context
        return model

    # ------------------------------------------------------------------
    # parsing / conformity
    # ------------------------------------------------------------------

    def _dynspec_variables(self) -> Dict[str, List[str]]:
        return {
            group: list(names)
            for group, names in self.symbols.items()
            if group != "parameters"
        }

    def _parse_group_trees(self) -> Dict[str, List[Any]]:
        """Parse raw equation strings into dynspec trees (with normalization).

        Complementarities (`res ⟂ lb <= x[t] <= ub`) are parsed natively by
        dynospec: the residual stays the equation, the bounds become the
        `*_lb` / `*_ub` bound-tree groups.  The legacy dolo timing
        conveniences (untimed symbols, `x(1)`, `==`) are still normalized at
        text level — they are not part of dynospec.
        """
        trees: Dict[str, List[Any]] = {}
        # Variable (non-parameter) symbols are the only names that get timed.
        variable_symbols = {
            n for g, names in self.symbols.items() if g != "parameters" for n in names
        }
        controls = list(self.symbols["controls"])

        for group, lines in self.__equations_text__.items():
            out = []
            for line in lines:
                if group in ("controls_lb", "controls_ub"):
                    # standalone bounds: `c = 0.1` style lines, keep the rhs
                    if "=" in line:
                        line = line.split("=", 1)[1].strip()
                out.append(_pre.normalize_timing(line, variable_symbols))

            bound_lb: List[Any] = []
            bound_ub: List[Any] = []
            group_trees: List[Any] = []
            any_bound = False
            for i, t in enumerate(_parse_block("\n".join(out))):
                comp = _complementarity_parts(t)
                if comp is None:
                    group_trees.append(t)
                    bound_lb.append(None)
                    bound_ub.append(None)
                    continue
                residual, lb, xname, ub = comp
                if group == "arbitrage" and controls and i < len(controls):
                    if xname != controls[i]:
                        raise Exception(
                            f"Complementarity condition in `{group}` equation "
                            f"{i + 1} is incorrect. Expected '{controls[i]}' "
                            f"instead of '{xname}'."
                        )
                any_bound = True
                group_trees.append(residual)
                bound_lb.append(lb)
                bound_ub.append(ub)

            trees[group] = group_trees
            if any_bound:
                # pad non-complementarity equations with -inf/inf bounds so
                # that there is one bound per equation (dolo convention)
                lb_trees = [t if t is not None else _neg_inf_tree() for t in bound_lb]
                ub_trees = [t if t is not None else _inf_tree() for t in bound_ub]
                self.__bound_trees__[f"{group}_lb"] = lb_trees
                self.__bound_trees__[f"{group}_ub"] = ub_trees
                self.__bound_text__[f"{group}_lb"] = [
                    str_expression(t) for t in lb_trees
                ]
                self.__bound_text__[f"{group}_ub"] = [
                    str_expression(t) for t in ub_trees
                ]
        return trees

    @property
    def variables(self) -> List[str]:
        return [
            n for g, names in self.symbols.items() if g != "parameters" for n in names
        ]

    @property
    def equations(self) -> Dict[str, List[str]]:
        res = {g: list(lines) for g, lines in self.__equations_text__.items()}
        for bname in ("arbitrage_lb", "arbitrage_ub"):
            if bname in self.__bound_text__:
                # make sure bounds text exists even before tree parsing
                self._ensure_parsed()
                res[bname] = list(self.__bound_text__[bname])
        return res

    def _ensure_parsed(self) -> None:
        if self.__equations_trees__ is None:
            self.__equations_trees__ = self._parse_group_trees()

    @property
    def definitions(self) -> Dict[str, str]:
        return dict(self.__definitions_text__)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------

    def get_calibration(self) -> Dict[str, float]:
        solved = _solve_symbolic_dict(
            self.__calibration_data__, self.definitions, self.symbols
        )
        for group, names in self.symbols.items():
            if group == "parameters":
                continue
            default = DEFAULT_INITIAL_VALUES.get(group, np.nan)
            for name in names:
                solved.setdefault(name, default)
        return solved

    @property
    def calibration(self) -> CalibrationDict:
        if self.__calibration__ is None:
            solved = self.get_calibration()
            grouped = _calibration_grouped(self.symbols, solved)
            self.__calibration__ = CalibrationDict(self.symbols, grouped)
        return self.__calibration__

    def set_calibration(self, *pargs, **kwargs) -> None:
        if len(pargs) == 1:
            return self.set_calibration(**pargs[0])
        self.__calibration_data__.update(kwargs)
        self.__calibration__ = None
        self.__exogenous__ = None

    @property
    def options(self) -> Dict[str, Any]:
        if self.__options__ is None:
            node = self.__options_node__
            if isinstance(node, dict):
                self.__options__ = node
            else:
                self.__options__ = _node_to_native(node) if node is not None else {}
        return self.__options__

    # ------------------------------------------------------------------
    # exogenous process
    # ------------------------------------------------------------------

    @property
    def exogenous(self):
        if self.__exogenous_object__ is None:
            return None
        if self.__exogenous__ is None:
            self.__exogenous__ = self.get_exogenous()
        return self.__exogenous__

    def get_exogenous(self):
        source = self.__exogenous_object__
        # YAML path: source is a (possibly tagged) node tree
        if not isinstance(source, (processes.Process, dict, list)):
            source = eval_data(source, self.get_calibration())
        if isinstance(source, processes.ProductProcess):
            return source
        if isinstance(source, processes.Process):
            return source
        if isinstance(source, list):
            return processes.ProductProcess(*source)
        # new style: dict of processes keyed by shock names
        syms = list(self.symbols["exogenous"])
        ssyms: List[str] = []
        for key in source.keys():
            ssyms.extend(v.strip() for v in str(key).split(","))
        if tuple(syms) != tuple(ssyms):
            raise Exception(
                "'exogenous' section. Shocks specification must match "
                f"declaration order. Found {tuple(ssyms)}. Expected {tuple(syms)}"
            )
        return processes.ProductProcess(*source.values())

    # ------------------------------------------------------------------
    # domain / grid
    # ------------------------------------------------------------------

    def get_domain(self):
        domain_data = self.__domain__
        if domain_data is None or len(domain_data) == 0:
            return None
        # YAML path: convert the node tree (expressions stay as strings)
        if not isinstance(domain_data, dict):
            domain_data = _node_to_native(domain_data)
        states = list(self.symbols["states"])
        sdomain = {k: v for k, v in domain_data.items() if k in states}
        if len(sdomain) < len(states):
            missing = [s for s in states if s not in sdomain]
            raise Exception(f"Missing domain for states: {', '.join(missing)}.")
        solved = self.get_calibration()
        evaluated = {}
        for name, bounds in sdomain.items():
            vals = []
            for b in bounds:
                if isinstance(b, (int, float)):
                    vals.append(b)
                else:
                    vals.append(_eval_scalar(str(b), solved))
            evaluated[name] = vals
        return CartesianDomain(**evaluated)

    @property
    def domain(self):
        return self.get_domain()

    @property
    def endo_grid(self):
        domain = self.get_domain()
        if domain is None:
            raise Exception("Model has no `domain` or `options:grid` specification.")
        options = self.options
        grid_section = (
            options.get("grid", {})
            if isinstance(options.get("grid"), (dict, TaggedDict))
            else {}
        )
        grid_type = grid_section.get("type")
        if grid_type is None and isinstance(grid_section, TaggedDict):
            grid_type = grid_section.tag.strip("!")
        grid_type = grid_type or options.get("grid_type")
        if grid_type is None:
            grid_type = "cartesian"
        grid_type = str(grid_type).lower()
        solved = self.get_calibration()
        if grid_type in ("cartesian", "cartesiangrid"):
            from dolo.numeric.grids import UniformCartesianGrid

            orders = grid_section.get("n") or grid_section.get("orders")
            if orders is None:
                orders = [20] * len(domain.min)
            return UniformCartesianGrid(min=domain.min, max=domain.max, n=orders)
        elif grid_type in ("nonuniformcartesian", "nonuniformcartesiangrid"):
            from dolo.numeric.grids import NonUniformCartesianGrid

            nodes = [
                _eval_scalar(str(e), solved) for e in grid_section.get("nodes", [])
            ]
            return NonUniformCartesianGrid(nodes)
        elif grid_type in ("smolyak", "smolyakgrid"):
            from dolo.numeric.grids import SmolyakGrid

            mu = grid_section.get("mu", 2)
            return SmolyakGrid(min=domain.min, max=domain.max, mu=mu)
        raise Exception("Unknown grid type.")

    def discretize(self, grid_options=None, dprocess_options={}):
        if self.exogenous is None:
            raise NotImplementedError("discretize() requires an `exogenous` section.")
        dprocess = self.exogenous.discretize(**dprocess_options)
        if grid_options is None:
            endo_grid = self.endo_grid
        else:
            endo_grid = self.domain.discretize(**grid_options)
        from dolo.numeric.grids import ProductGrid

        grid = ProductGrid(dprocess.grid, endo_grid, names=["exo", "endo"])
        return [grid, dprocess]

    # ------------------------------------------------------------------
    # functions
    # ------------------------------------------------------------------

    def _get_definition_trees(self) -> List[Any]:
        variable_symbols = {
            n for g, names in self.symbols.items() if g != "parameters" for n in names
        }
        lines = []
        for lhs, rhs in self.__definitions_text__.items():
            lhs_n = _pre.normalize_timing(str(lhs), variable_symbols)
            rhs_n = _pre.normalize_timing(str(rhs), variable_symbols)
            lines.append(f"{lhs_n} = {rhs_n}")
        return _parse_block("\n".join(lines))

    def _build_functions(self) -> LoosyDict:
        self._ensure_parsed()
        trees = self.__equations_trees__
        variables = self._dynspec_variables()
        parameters = list(self.symbols["parameters"])

        definitions_trees = self._get_definition_trees()
        self.__defs_block__ = None
        if definitions_trees:
            _, self.__defs_block__ = functions_v2.build_definitions(
                equations=definitions_trees,
                variables=variables,
                parameters=parameters,
            )

        functions = LoosyDict(
            equivalences={
                "equilibrium": "arbitrage",
                "optimality": "arbitrage",
                "controls_lb": "arbitrage_lb",
                "controls_ub": "arbitrage_ub",
            }
        )
        for fname, eqs in trees.items():
            if fname not in functions_v2.GROUP_LAYOUTS:
                raise NotImplementedError(
                    f"Equation group `{fname}` is not supported by ModelV2 yet."
                )
            layout = functions_v2.GROUP_LAYOUTS[fname]
            if fname == "arbitrage" or layout.with_definitions:
                defs = self.__defs_block__
            else:
                defs = None
            functions[fname] = functions_v2.compile_group(
                name=fname,
                equations=eqs,
                variables=variables,
                parameters=parameters,
                definitions_block=defs,
            )
        if definitions_trees:
            functions["auxiliary"] = functions_v2.compile_group(
                name="auxiliary",
                equations=definitions_trees,
                variables=variables,
                parameters=parameters,
                definitions_block=None,
            )
        for bname in ("arbitrage_lb", "arbitrage_ub"):
            if bname in self.__bound_trees__:
                functions[bname] = functions_v2.compile_group(
                    name=bname,
                    equations=self.__bound_trees__[bname],
                    variables=variables,
                    parameters=parameters,
                    definitions_block=None,
                )
        return functions

    @property
    def functions(self) -> LoosyDict:
        if self.__functions__ is None:
            self.__functions__ = self._build_functions()
            # legacy algorithm internals (`egm`) read these attributes
            self.__original_functions__ = self.__functions__
            self.__original_gufunctions__ = self.__functions__
        return self.__functions__

    @property
    def x_bounds(self):
        if "controls_ub" in self.functions:
            return [self.functions["controls_lb"], self.functions["controls_ub"]]
        if "arbitrage_ub" in self.functions:
            return [self.functions["arbitrage_lb"], self.functions["arbitrage_ub"]]
        return None

    # ------------------------------------------------------------------
    # convenience
    # ------------------------------------------------------------------

    @property
    def infos(self) -> Dict[str, Any]:
        return {"name": self.name, "filename": self.filename, "type": self.model_type}

    def residuals(self, calib=None):
        from dolo.algos.steady_state import residuals

        return residuals(self, calib)

    def eval_formula(self, expr, dataframe=None, calib=None):
        """Evaluate a formula with dolo timing notation (``k(0)``, ``k(-1)``).

        Untimed names are resolved against the calibration; timed names
        (``x(±d)``) are looked up as columns of `dataframe`, shifted by
        `d` rows (`x(1)` = one period back, like the legacy evaluator).
        ``k[t±d]`` (dynspec) timing is accepted as well.  Built on
        ``dyno.dynspec`` — no dolang involved.
        """
        if calib is None:
            calib = self.calibration
        context = calib.flat if isinstance(calib, CalibrationDict) else dict(calib)

        columns = set(dataframe.columns) if dataframe is not None else set()
        variable_names = (
            {n for g, names in self.symbols.items() if g != "parameters" for n in names}
            | set(self.definitions)
            | set(columns)
        )

        expr = _pre.rewrite_timing_calls(str(expr), variable_names)
        tree = parser.parse(expr, start="formula")

        variables: Dict[str, Dict[int, Any]] = {}
        if dataframe is not None:
            for name, shift in _timed_variables(tree):
                variables.setdefault(name, {})[shift] = dataframe[name].shift(shift)
            import pandas as pd

            context = dict(context)
            context["t_"] = pd.Series(dataframe.index, index=dataframe.index)

        ev = FormulaEvaluator(
            {
                "constants": dict(context),
                "variables": variables,
                "values": {},
                "processes": {},
            }
        )
        return ev.visit(tree)

    def __str__(self):
        from dolo.misc.termcolor import colored

        s = (
            f'Model:\n------\nname: "{self.name}"\n'
            f'type: "{self.model_type}"\nfile: "{self.filename}"\n'
        )
        ss = "\nEquations:\n----------\n\n"
        res = self.residuals() if self.functions else {}

        equations = self.equations.copy()
        if self.definitions:
            equations["definitions"] = [
                f"{lhs} = {rhs}" for lhs, rhs in self.definitions.items()
            ]
        for group, eqs in equations.items():
            if group in ("arbitrage_lb", "arbitrage_ub"):
                continue  # bounds are shown embedded in the arbitrage lines
            vals_group = res.get(group)
            ss += f"{group}\n"
            for i, eq in enumerate(eqs):
                val = (
                    vals_group[i]
                    if vals_group is not None and i < len(vals_group)
                    else None
                )
                if val is None:
                    ss += f" {i+1:2} : {eq}\n"
                else:
                    if abs(val) < 1e-8:
                        val = 0.0
                    vals = f"{val:.4f}"
                    if abs(val) > 1e-8:
                        vals = colored(vals, "red")
                    ss += f" {i+1:2} : {vals} : {eq}\n"
            ss += "\n"
        return s + ss

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        # Equations rendered with dynspec latex (`x[t-1]` -> `x_{t-1}`,
        # greek names -> greek commands) — no dolang involved.
        from dyno.dynspec.latex import latex

        def _tex(eq: str) -> str:
            eq = eq.strip()
            for sep in ("⟂", "|"):
                if sep in eq:
                    eq = eq.split(sep, 1)[0].strip()
            if "=" in eq:
                lhs, rhs = [e.strip() for e in eq.split("=", 1)]
                return f"{latex(lhs)} = {latex(rhs)}"
            return latex(eq)

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

        resids = self.residuals() if self.functions else {}
        equations = self.equations.copy()
        if self.definitions:
            equations["definitions"] = [
                f"{lhs} = {rhs}" for lhs, rhs in self.definitions.items()
            ]

        table = (
            '<tr><td><b>Type</b></td><td style="width:80%"><b>Equation</b></td>'
            "<td><b>Residual</b></td></tr>\n"
        )
        for eq_type, eqs in equations.items():
            if eq_type in ("arbitrage_lb", "arbitrage_ub"):
                continue  # bounds are shown embedded in the arbitrage lines
            for i, eq in enumerate(eqs):
                if eq_type in ("arbitrage", "transition", "arbitrage_exp"):
                    val = resids[eq_type][i]
                    if abs(val) > 1e-8:
                        vals = '<span style="color: red;">{:.4f}</span>'.format(val)
                    else:
                        vals = "{:.3f}".format(val)
                else:
                    vals = ""
                try:
                    lat = "${}$".format(_tex(eq))
                except Exception:
                    lat = str(eq).replace("<", "&lt").replace(">", "&gt")
                h = eq_type if i == 0 else ""
                fmt_line = "<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    h, lat, vals
                )
                table += fmt_line + "\n"
        table = "<table>{}</table>".format(table)

        return table_infos + table


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def extract_variables_in(tree) -> set:
    from dyno.dynspec.recipe import extract_variables_from_equation

    return set(extract_variables_from_equation(tree).keys())


def _name_of(node) -> Optional[str]:
    if node is None:
        return None
    from yaml import ScalarNode

    if isinstance(node, ScalarNode):
        return str(_scalar_value(node))
    return str(node)


def _normalize_definitions_nodes(definitions: Any) -> Dict[str, str]:
    """Normalize the `definitions` YAML node into {aux: rhs-string}."""
    from yaml import MappingNode, ScalarNode, SequenceNode

    if definitions is None:
        return {}
    if isinstance(definitions, ScalarNode):
        defs: Dict[str, str] = {}
        for line in _pre.split_equations(definitions.value):
            if "=" not in line:
                raise Exception(f"Invalid definition line: {line}")
            lhs, rhs = [e.strip() for e in line.split("=", 1)]
            defs[_strip_timing(lhs)] = rhs
        return defs
    if isinstance(definitions, MappingNode):
        return {
            _strip_timing(_key_text(k)): str(_scalar_value(v))
            for k, v in definitions.items()
        }
    if isinstance(definitions, SequenceNode):
        out: Dict[str, str] = {}
        for item in definitions:
            if isinstance(item, MappingNode):
                pair = list(item.items())
                out[_strip_timing(_key_text(pair[0][0]))] = str(
                    _scalar_value(pair[0][1])
                )
            else:
                out[_strip_timing(_key_text(item[0]))] = str(_scalar_value(item[1]))
        return out
    raise Exception(f"Unexpected definitions section: {definitions}")


def _strip_timing(name: str) -> str:
    import re

    return re.sub(r"\[t([+-]?\d*)\]", "", str(name)).strip()


def _parse_block(text: str) -> List[Any]:
    text = text.strip()
    if not text:
        return []
    tree = parser.parse(text, start="equation_block")
    return list(tree.children)


def _complementarity_parts(tree) -> Optional[tuple[Any, Any, str, Any]]:
    """Split a dynspec `complementarity` tree into its parts.

    Returns ``(residual, lb, middle_name, ub)``, or ``None`` for plain
    equations.  The dynospec grammar normalizes every complementarity to
    ``complementarity(residual, lb, x, ub)`` with explicit `-inf`/`inf`
    bounds (`⟂` and legacy `|` separators).
    """
    if getattr(tree, "data", None) != "complementarity":
        return None
    res, lb, x, ub = tree.children
    name = str(x.children[0].children[0])
    return res, lb, name, ub


def _inf_tree():
    """A dynspec tree for the built-in constant `inf` (bounds padding)."""
    from lark import Tree, Token

    return Tree("constant", [Tree("name", [Token("NAME", "inf")])])


def _neg_inf_tree():
    """A dynspec tree for `-inf`."""
    from lark import Tree

    return Tree("neg", [_inf_tree()])


def _tree_text(tree) -> str:
    """`str_expression` output for a non-annotated tree — but bare_formula
    trees come back as token lists; normalise either way."""
    res = str_expression(tree)
    return "".join(res) if isinstance(res, list) else str(res)


def _equation_rhs_text(eq: Any) -> str:
    if getattr(eq, "data", None) == "equality":
        return _tree_text(eq.children[1])
    return _tree_text(eq)


def _convert_processes(processes_dict: Dict) -> Any:
    """Convert dyno processes into dolo process objects."""
    from dyno.dynspec.language import Normal as DynoNormal

    if not processes_dict:
        return None
    converted = []
    for names, process in processes_dict.items():
        if isinstance(process, DynoNormal):
            Sigma = np.array(process.Σ, dtype=float)
            Μ = np.asarray(process.Μ, dtype=float).ravel()
            converted.append(distribution.Normal(Σ=Sigma, Μ=Μ))
        else:
            raise NotImplementedError(
                f"Converting dyno process {process.__class__.__name__} "
                "to a dolo process is not implemented yet."
            )
    if len(converted) == 1:
        return converted[0]
    return processes.ProductProcess(*converted)
