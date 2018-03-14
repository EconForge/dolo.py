# plainformatter = get_ipython().display_formatter.formatters['text/plain']
# del plainformatter.type_printers[dict]

import yaml
import numpy as np
from typing import List

import ast
from ast import BinOp, Sub

from typing import Dict

import dolang
from dolang import to_source, parse_string
from dolang.symbolic import time_shift
from dolang.symbolic import ExpressionSanitizer
from dolang.factory import FlatFunctionFactory

def get_name(e):
    return ast.parse(e).body[0].value.func.id

def reorder_preamble(pr):
    import sympy
    from dolang.triangular_solver import solve_triangular_system
    unknowns = [*pr.keys()]
    incidence = [[str(e) for e in sympy.sympify((exp)).atoms()] for exp in pr.values()]
    sol = solve_triangular_system(dict(zip(unknowns, incidence)))
    return dict([(k, pr[k]) for k in sol.keys()])

def get_factory(model, eq_type: str):

    from dolo.compiler.recipes import recipes
    from dolang.symbolic import stringify, stringify_symbol
    import re

    specs = recipes['dtcc']['specs'][eq_type]

    # this is acutally incorrect

    preamble_tshift = set([s[1] for s in specs['eqs'] if s[0]=='states'])
    preamble_tshift = preamble_tshift.intersection(set([s[1] for s in specs['eqs'] if s[0]=='controls']))

    args = []
    for sg in specs['eqs']:
        if sg[0] == 'parameters':
            args.append([s for s in model.symbols["parameters"]])
        else:
            args.append([(s, sg[1]) for s in model.symbols[sg[0]]])
    args = [[stringify_symbol(e) for e in vg] for vg in args]

    arguments = dict( zip([sg[2] for sg in specs['eqs']], args) )

    eqs = model.equations[eq_type]

    if 'target' in specs:
        sg = specs['target']
        targets = [(s, sg[1]) for s in model.symbols[sg[0]]]
    else:
        targets = [('out{}'.format(i),0) for i in range(len(eqs))]

    regex = re.compile(r"\s*([^\|\s][^\|]*[^\|\s])\s*(\|([^\|\s][^\|]*[^\|\s])|\s*)")
    eqs = [regex.match(eq).group(1) for eq in eqs]

    if 'target' in specs:
        eqs = [eq.split('=')[1] for eq in eqs]
    else:
        eqs = [("({1})-({0})".format(*eq.split('=')) if '=' in eq else eq) for eq in eqs]

    eqs = [str.strip(eq) for eq in eqs]
    eqs = [dolang.parse_string(eq) for eq in eqs]
    es = ExpressionSanitizer(model.variables)
    eqs = [es.visit(eq) for eq in eqs]
    eqs = [stringify(eq) for eq in eqs]
    eqs = [dolang.to_source(eq) for eq in eqs]

    targets = [stringify_symbol(e) for e in targets]

    # sanitize defs ( should be )
    defs = dict()
    for k in model.definitions:
        if '(' not in k:
            s = "{}(0)".format(k)
            val = model.definitions[k]
            val = es.visit(dolang.parse_string(val))
            for t in preamble_tshift:
                s = stringify_symbol((k,t))
                vv = stringify(time_shift(val, t))
                defs[s] = dolang.to_source(vv)

    preamble = reorder_preamble(defs)

    eqs = dict(zip(targets, eqs))
    ff = FlatFunctionFactory(preamble, eqs, arguments, eq_type)

    return ff
