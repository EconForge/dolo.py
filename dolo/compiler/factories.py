# plainformatter = get_ipython().display_formatter.formatters['text/plain']
# del plainformatter.type_printers[dict]

import yaml
import numpy as np
from typing import List

import ast
from ast import BinOp, Sub

from typing import Dict

import dolang
from dolang.grammar import str_expression
from dolang.symbolic import parse_string
from dolang.symbolic import time_shift
from dolang.symbolic import Sanitizer
from dolang.factory import FlatFunctionFactory


def get_name(e):
    return ast.parse(e).body[0].value.func.id


def reorder_preamble(pr):

    from dolang.triangular_solver import triangular_solver, get_incidence

    inc = get_incidence(pr)
    order = triangular_solver(inc)
    d = dict()
    prl = [*pr.items()]
    for o in order:
        k, v = prl[o]
        d[k] = v
    return d


def shift_spec(specs, tshift):
    ss = dict()
    if "target" in specs:
        e = specs["target"]
        ss["target"] = [e[0], e[1] + tshift, e[2]]
    ss["eqs"] = [
        ([e[0], e[1] + tshift, e[2]] if e[0] != "parameters" else e)
        for e in specs["eqs"]
    ]
    return ss


def get_factory(model, eq_type: str, tshift: int = 0):

    from dolo.compiler.model import decode_complementarity

    from dolo.compiler.recipes import recipes
    from dolang.symbolic import stringify, stringify_symbol

    equations = model.equations

    if eq_type == "auxiliary":
        eqs = ["{}".format(s) for s in model.symbols["auxiliaries"]]
        specs = {
            "eqs": [
                ["exogenous", 0, "m"],
                ["states", 0, "s"],
                ["controls", 0, "x"],
                ["parameters", 0, "p"],
            ]
        }
    else:
        eqs = equations[eq_type]
        if eq_type in ("arbitrage_lb", "arbitrage_ub"):
            specs = {
                "eqs": recipes["dtcc"]["specs"]["arbitrage"]["complementarities"][
                    "left-right"
                ]
            }
        else:
            specs = recipes["dtcc"]["specs"][eq_type]

    specs = shift_spec(specs, tshift=tshift)

    preamble_tshift = set([s[1] for s in specs["eqs"] if s[0] == "states"])
    preamble_tshift = preamble_tshift.intersection(
        set([s[1] for s in specs["eqs"] if s[0] == "controls"])
    )

    args = []
    for sg in specs["eqs"]:
        if sg[0] == "parameters":
            args.append([s for s in model.symbols["parameters"]])
        else:
            args.append([(s, sg[1]) for s in model.symbols[sg[0]]])
    args = [[stringify_symbol(e) for e in vg] for vg in args]

    arguments = dict(zip([sg[2] for sg in specs["eqs"]], args))

    # temp
    eqs = [eq.split("⟂")[0].strip() for eq in eqs]

    if "target" in specs:
        sg = specs["target"]
        targets = [(s, sg[1]) for s in model.symbols[sg[0]]]
        eqs = [eq.split("=")[1] for eq in eqs]
    else:
        eqs = [
            ("({1})-({0})".format(*eq.split("=")) if "=" in eq else eq) for eq in eqs
        ]
        targets = [("out{}".format(i), 0) for i in range(len(eqs))]

    eqs = [str.strip(eq) for eq in eqs]

    eqs = [dolang.parse_string(eq) for eq in eqs]
    es = Sanitizer(variables=model.variables)
    eqs = [es.transform(eq) for eq in eqs]

    eqs = [time_shift(eq, tshift) for eq in eqs]

    eqs = [stringify(eq) for eq in eqs]

    eqs = [str_expression(eq) for eq in eqs]

    targets = [stringify_symbol(e) for e in targets]

    # sanitize defs ( should be )
    defs = dict()

    for k in model.definitions:
        val = model.definitions[k]
        # val = es.transform(dolang.parse_string(val))
        for t in preamble_tshift:
            s = stringify(time_shift(k, t))
            if isinstance(val, str):
                vv = stringify(time_shift(val, t))
            else:
                vv = str(val)
            defs[s] = vv

    preamble = reorder_preamble(defs)

    eqs = dict(zip(targets, eqs))
    ff = FlatFunctionFactory(preamble, eqs, arguments, eq_type)

    return ff
