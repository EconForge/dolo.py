from __future__ import division

import copy


def triangular_solver(incidence, context=None):

    if context is None:

        context = dict()
        import math
        context['log'] = math.log
        context['exp'] = math.exp
        context['nan'] = float('nan')

    n = len(incidence)

    current = copy.deepcopy(incidence)

    solved = []
    max_steps = len(incidence)
    steps = 0

    while (steps < max_steps) and len(solved) < n:

        possibilities = [i for i in range(n) if (
            i not in solved) and (len(current[i]) == 0)]

        for i in possibilities:
            for e in current:
                if i in e:
                    e.remove(i)
            solved.append(i)

        steps += 1

    if len(solved) < n:
        raise Exception('System is not triangular')
    return solved


def get_incidence(sdict):

    var_order = list(sdict.keys())
    var_set = set(var_order)
    expressions = [sdict[k] for k in var_order]
    incidence = []

    for i, eq in enumerate(expressions):

        atoms = get_atoms(eq)
        vars = var_set.intersection(atoms)
        inds = [var_order.index(v) for v in vars]
        incidence.append(inds)

    return incidence


from collections import OrderedDict


def solve_triangular_system(system, values=None, context=None):

    system = OrderedDict(system)

    var_order = list(system.keys())

    ll = get_incidence(system)

    sol_order = triangular_solver(ll, context=context)

    d = copy.copy(values) if values else {}


    import math
    d['nan'] = float('nan')
    d['exp'] = math.exp
    d['log'] = math.log
    d['sin'] = math.sin
    d['cos'] = math.cos

    for i in sol_order:

        v = var_order[i]

        try:

            val = system[v]

            d[v] = eval(str(val), d, d)

        except Exception as e:  # in case d[v] is an int

            raise(e)

    resp = OrderedDict([(v, d[v]) for v in system.keys()])

    return resp


import ast
from ast import NodeVisitor


def get_atoms(string):

    expr = ast.parse(str.strip(str(string)))
    parser = FindNames()
    parser.visit(expr)
    names = parser.names

    return set(names)


class FindNames(NodeVisitor):

    def __init__(self):
        self.names = []

    def visit_Name(self, node):
        self.names.append(node.id)
