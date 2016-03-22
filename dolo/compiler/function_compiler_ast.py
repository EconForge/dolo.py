
from __future__ import division
from dolo.compiler.codegen import to_source
from numba import njit, guvectorize
import copy

import sys
is_python_3 = sys.version_info >= (3, 0)


def to_expr(s):
    import ast
    if isinstance(s, ast.Expr):
        return copy.deepcopy(s)
    else:
        return ast.parse(s).body[0].value

def std_date_symbol(s, date):
    if date == 0:
        return '{}_'.format(s)
    elif date <= 0:
        return '{}_m{}_'.format(s, str(-date))
    elif date >= 0:
        return '{}__{}_'.format(s, str(date))


import ast

from ast import Expr, Subscript, Name, Load, Index, Num, UnaryOp, UAdd, Module, Assign, Store, Call, Module, FunctionDef, arguments, Param, ExtSlice, Slice, Ellipsis, Call, Str, keyword, NodeTransformer, Tuple, USub

# def Name(id=id, ctx=None): return ast.arg(arg=id)

class TimeShiftTransformer(ast.NodeTransformer):
    def __init__(self, variables, shift=0):

        self.variables = variables
        self.shift = shift

    def visit_Name(self, node):
        name = node.id
        if name in self.variables:
            if self.shift==0 or self.shift=='S':
                return ast.parse(name).body[0].value
            else:
                return ast.parse('{}({})'.format(name,self.shift)).body[0].value
        else:
             return node

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            if self.shift =='S':
                return ast.parse('{}'.format(name)).body[0].value
            else:
                new_date = date+self.shift
                if new_date != 0:
                    return ast.parse('{}({})'.format(name,new_date)).body[0].value
                else:
                    return ast.parse('{}'.format(name)).body[0].value
        else:

            # , keywords=node.keywords,  kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])

import copy
def timeshift(expr, variables, shift):
    if isinstance(expr, str):
        aexpr = ast.parse(expr).body[0].value
    else:
        aexpr = copy.deepcopy(expr)
    resp = TimeShiftTransformer(variables, shift).visit(aexpr)
    if isinstance(expr, str):
        return to_source(resp)
    else:
        return resp

class StandardizeDatesSimple(NodeTransformer):

    # replaces calls to variables by time subscripts

    def __init__(self, tvariables):

        self.tvariables = tvariables  # list of variables
        self.variables = [e[0] for e in tvariables]
        # self.variables = tvariables # ???

    def visit_Name(self, node):

        name = node.id
        newname = std_date_symbol(name, 0)
        if (name, 0) in self.tvariables:
            expr = Name(newname, Load())
            return expr
        else:
            return node

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            newname = std_date_symbol(name, date)
            if newname is not None:
                return Name(newname, Load())

        else:

            # , keywords=node.keywords, starargs=node.starargs, kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])


class StandardizeDates(NodeTransformer):

    def __init__(self, symbols, arg_names):

        table = {}
        for a in arg_names:
            t = tuple(a)
            symbol_group = a[0]
            date = a[1]
            an = a[2]

            for b in symbols[symbol_group]:
                index = symbols[symbol_group].index(b)
                table[(b, date)] = (an, date)

        variables = [k[0] for k in table]

        table_symbols = {k: (std_date_symbol(*k)) for k in table.keys()}

        self.table = table
        self.variables = variables  # list of vari
        self.table_symbols = table_symbols

    def visit_Name(self, node):

        name = node.id
        key = (name, 0)
        if key in self.table:
            newname = self.table_symbols[key]
            expr = Name(newname, Load())
            return expr
        else:
            return node

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]
        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            key = (name, date)
            newname = self.table_symbols.get(key)
            if newname is not None:
                return Name(newname, Load())
            else:
                raise Exception(
                    "Symbol {} incorrectly subscripted with date {}.".format(name, date))
        else:

            # , keywords=node.keywords,  kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])


class ReplaceName(ast.NodeTransformer):

    # replaces names according to definitions

    def __init__(self, defs):
        self.definitions = defs

    def visit_Name(self, expr):
        if expr.id in self.definitions:
            return self.definitions[expr.id]
        else:
            return expr


def compile_function_ast(expressions, symbols, arg_names, output_names=None, funname='anonymous', return_ast=False, print_code=False, definitions=None, vectorize=True, use_file=False):
    '''
    expressions: list of equations as string
    '''

    from collections import OrderedDict
    table = OrderedDict()

    aa = arg_names

    if output_names is not None:
        aa = arg_names + [output_names]

    for a in aa:
        symbol_group = a[0]
        date = a[1]
        an = a[2]

        for b in symbols[symbol_group]:
            index = symbols[symbol_group].index(b)
            table[(b, date)] = (an, index)

    table_symbols = {k: (std_date_symbol(*k)) for k in table.keys()}

    # standard assignment: i.e. k = s[0]
    index = lambda x: Index(Num(x))

    # declare symbols

    aux_short_names = [e[2] for e in arg_names if e[0]=='auxiliaries']



    preamble = []

    for k in table:  # order it
        # k : var, date
        arg, pos = table[k]
        if not (arg in aux_short_names):
            std_name = table_symbols[k]
            val = Subscript(value=Name(id=arg, ctx=Load()), slice=index(pos), ctx=Load())
            line = Assign(targets=[Name(id=std_name, ctx=Store())], value=val)
            if arg != 'out':
                preamble.append(line)

    body = []
    std_dates = StandardizeDates(symbols, aa)


    if definitions is not None:
        for k,v in definitions.items():
            if isinstance(k, str):
                lhs = ast.parse(k).body[0].value
            if isinstance(v, str):
                rhs = ast.parse(v).body[0].value
            else:
                rhs = v
            lhs = std_dates.visit(lhs)
            rhs = std_dates.visit(rhs)
            vname = lhs.id
            line = Assign(targets=[Name(id=vname, ctx=Store())], value=rhs)
            preamble.append(line)


    outs = []
    for i, expr in enumerate(expressions):

        expr = ast.parse(expr).body[0].value
        # if definitions is not None:
        #     expr = ReplaceName(defs).visit(expr)

        rexpr = std_dates.visit(expr)

        rhs = rexpr

        if output_names is not None:
            varname = symbols[output_names[0]][i]
            date = output_names[1]
            out_name = table_symbols[(varname, date)]
        else:
            out_name = 'out_{}'.format(i)

        line = Assign(targets=[Name(id=out_name, ctx=Store())], value=rhs)
        body.append(line)

        line = Assign(targets=[Subscript(value=Name(id='out', ctx=Load()),
                                         slice=index(i), ctx=Store())], value=Name(id=out_name, ctx=Load()))
        body.append(line)

    arg_names = [e for e in arg_names if e[0]!="auxiliaries"]

    args = [e[2] for e in arg_names] + ['out']

    if is_python_3:
        from ast import arg
        f = FunctionDef(name=funname, args=arguments(args=[arg(arg=a) for a in args], vararg=None, kwarg=None, kwonlyargs=[], kw_defaults=[], defaults=[]),
                        body=preamble + body, decorator_list=[])
    else:
        f = FunctionDef(name=funname, args=arguments(args=[Name(id=a, ctx=Param()) for a in args], vararg=None, kwarg=None, kwonlyargs=[], kw_defaults=[], defaults=[]),
                        body=preamble + body, decorator_list=[])

    mod = Module(body=[f])
    mod = ast.fix_missing_locations(mod)

    if print_code:
        s = "Function {}".format(mod.body[0].name)
        print("-" * len(s))
        print(s)
        print("-" * len(s))
        print(to_source(mod))

    if vectorize:
        from numba import float64, void
        coredims = [len(symbols[an[0]]) for an in arg_names]
        signature = str.join(',', ['(n_{})'.format(d) for d in coredims])
        n_out = len(expressions)
        if n_out in coredims:
            signature += '->(n_{})'.format(n_out)
            # ftylist = float64[:](*([float64[:]] * len(coredims)))
            fty = "void(*[float64[:]]*{})".format(len(coredims)+1)
        else:
            signature += ',(n_{})'.format(n_out)
            fty = "void(*[float64[:]]*{})".format(len(coredims)+1)
        ftylist = [fty]
    else:
        signature=None
        ftylist=None

    if use_file:
        fun = eval_ast_with_file(mod, print_code=True)
    else:
        fun = eval_ast(mod)

    jitted = njit(fun)
    if vectorize:
        gufun = guvectorize([fty], signature, target='parallel', nopython=True)(fun)
        return jitted, gufun
    else:
        return jitted


def eval_ast(mod):

    context = {}

    context['division'] = division  # THAT seems strange !

    import numpy

    context['inf'] = numpy.inf
    context['maximum'] = numpy.maximum
    context['minimum'] = numpy.minimum

    context['exp'] = numpy.exp
    context['log'] = numpy.log
    context['sin'] = numpy.sin
    context['cos'] = numpy.cos

    context['abs'] = numpy.abs

    name = mod.body[0].name
    mod = ast.fix_missing_locations(mod)
    # print( ast.dump(mod) )
    code = compile(mod, '<string>', 'exec')
    exec(code, context, context)
    fun = context[name]

    return fun


def eval_ast_with_file(mod, print_code=False, signature=None, ftylist=None):

    name = mod.body[0].name

    code = """\
from __future__ import division

from numpy import exp, log, sin, cos, abs
from numpy import inf, maximum, minimum
"""

#     if signature is not None:
#         print(signature)
#
#         decorator = """
# from numba import float64, void, guvectorize
# @guvectorize(signature='{signature}', ftylist={ftylist}, target='parallel', nopython=True)
# """.format(signature=signature, ftylist=ftylist)
#         code += decorator

    code += to_source(mod)

    if print_code:
        print(code)

    import sys
    # try to create a new file
    import time
    import tempfile
    import os, importlib
    from dolo.config import temp_dir
    temp_file = tempfile.NamedTemporaryFile(mode='w+t', prefix='fun', suffix='.py', dir=temp_dir, delete=False)
    with temp_file:
        temp_file.write(code)
    modname = os.path.basename(temp_file.name).strip('.py')


    full_name = os.path.basename(temp_file.name)
    modname, extension = os.path.splitext(full_name)

    module = importlib.import_module(modname)

    fun =  module.__dict__[name]

    return fun


def test_compile_allocating():
    from collections import OrderedDict
    eq = ['(a + b*exp(p1))', 'p2*a+b']
    symtypes = [
        ['states', 0, 'x'],
        ['parameters', 0, 'p']
    ]
    symbols = OrderedDict([('states', ['a', 'b']),
                           ('parameters', ['p1', 'p2'])
                           ])
    gufun = compile_function_ast(eq, symbols, symtypes, data_order=None)
    n_out = len(eq)

    import numpy
    N = 1000
    vecs = [numpy.zeros((N, len(e))) for e in symbols.values()]
    out = numpy.zeros((N, n_out))
    gufun(*(vecs + [out]))


def test_compile_non_allocating():
    from collections import OrderedDict
    eq = ['(a + b*exp(p1))', 'p2*a+b', 'a+p1']
    symtypes = [
        ['states', 0, 'x'],
        ['parameters', 0, 'p']
    ]
    symbols = OrderedDict([('states', ['a', 'b']),
                           ('parameters', ['p1', 'p2'])
                           ])
    gufun = compile_function_ast(eq, symbols, symtypes, use_numexpr=False,
                                 data_order=None, vectorize=True)
    n_out = len(eq)

    import numpy
    N = 1000
    vecs = [numpy.zeros((N, len(e))) for e in symbols.values()]
    out = numpy.zeros((N, n_out))
    gufun(*(vecs + [out]))
    d = {}
    try:
        allocated = gufun(*vecs)
    except Exception as e:
        d['error'] = e
    if len(d) == 0:
        raise Exception("Frozen dimensions may have landed in numba ! Check.")
    # assert(abs(out-allocated).max()<1e-8)

if __name__ == "__main__":
    test_compile_allocating()
    test_compile_non_allocating()
    print("Done")
