
from __future__ import division
from .codegen import to_source



def std_date_symbol(s,date):
    if date == 0:
        return '{}_'.format(s)
    elif date <= 0:
        return '{}_m{}_'.format(s, str(-date))
    elif date >= 0:
        return '{}__{}_'.format(s, str(date))


import ast

from ast import Expr, Subscript, Name, Load, Index, Num, UnaryOp, UAdd, Module, Assign, Store, Call, Module, FunctionDef, arguments, Param, ExtSlice, Slice, Ellipsis, Call, Str, keyword, NodeTransformer

# def Name(id=id, ctx=None): return ast.arg(arg=id)


class StandardizeDatesSimple(NodeTransformer):

    # replaces calls to variables by time subscripts

    def __init__(self, tvariables):

        self.tvariables = tvariables # list of variables
        self.variables = [e[0] for e in tvariables]


    def visit_Name(self, node):

        name = node.id
        newname = std_date_symbol(name, 0)
        if (name,0) in self.tvariables:
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
                assert(isinstance(args.op, UAdd))
                args = args.operand
            date = args.n
            newname = std_date_symbol(name, date)
            if newname is not None:
                return Name(newname, Load())

        else:

            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=node.keywords, starargs=node.starargs, kwargs=node.kwargs)


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
                table[(b,date)] = (an, date)

        variables = [k[0] for k in table]

        table_symbols = { k: (std_date_symbol(*k)) for k in table.keys() }

        self.table = table
        self.variables = variables # list of vari
        self.table_symbols = table_symbols


    def visit_Name(self, node):

        name = node.id
        key = (name,0)
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
                assert(isinstance(args.op, UAdd))
                args = args.operand
            date = args.n
            key = (name, date)
            newname = self.table_symbols.get(key)
            if newname is not None:
                return Name(newname, Load())
            else:
                raise Exception("Symbol {} incorrectly subscripted with date {}.".format(name, date))
        else:

            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=node.keywords, starargs=node.starargs, kwargs=node.kwargs)

class ReplaceName(ast.NodeTransformer):

    # replaces names according to definitions

    def __init__(self, defs):
        self.definitions = defs

    def visit_Name(self, expr):
        if expr.id in self.definitions:
            return self.definitions[expr.id]
        else:
            return expr


def compile_function_ast(expressions, symbols, arg_names, output_names=None, funname='anonymous',
     data_order='columns', use_numexpr=False, return_ast=False, print_code=False, definitions=None):

    '''
    expressions: list of equations as string
    '''

    vectorization_type = 'ellipsis'
    data_order = 'columns'

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

            table[(b,date)] = (an, index)

    table_symbols = { k: (std_date_symbol(*k)) for k in table.keys() }

    if data_order is None:
        # standard assignment: i.e. k = s[0]
        index = lambda x: Index(Num(x))
    elif vectorization_type == 'ellipsis':
        el = Ellipsis()
        if data_order == 'columns':
            # column broadcasting: i.e. k = s[...,0]
            index = lambda x: ExtSlice(dims=[el, Index(value=Num(n=x))])
        else:
            # rows broadcasting: i.e. k = s[0,...]
            index = lambda x: ExtSlice(dims=[Index(value=Num(n=x)),el])

    # declare symbols

    preamble = []

    for k in table: # order it
        # k : var, date
        arg,pos = table[k]
        std_name = table_symbols[k]
        val = Subscript(value=Name(id=arg, ctx=Load()), slice=index(pos), ctx=Load())
        line = Assign(targets=[Name(id=std_name, ctx=Store())], value=val)
        preamble.append(line)

    if use_numexpr:
        for i in range(len(expressions)):
        # k : var, date
            val = Subscript(value=Name(id='out', ctx=Load()), slice=index(i), ctx=Load())
            line = Assign(targets=[Name(id='out_{}'.format(i), ctx=Store())], value=val)
            preamble.append(line)

    body = []
    std_dates = StandardizeDates(symbols, aa)

    if definitions is not None:
        defs = {e: ast.parse(definitions[e]).body[0].value for e in definitions}

    for i,expr in enumerate(expressions):

        expr = ast.parse(expr).body[0].value
        if definitions is not None:
            expr = ReplaceName(defs).visit(expr)

        rexpr = std_dates.visit(expr)

        if not use_numexpr:
            rhs = rexpr
        else:
            src = to_source(rexpr)
            rhs = Call( func=Name(id='evaluate', ctx=Load()),
                args=[Str(s=src)], keywords=[keyword(arg='out', value=Name(id='out_{}'.format(i), ctx=Load()))], starargs=None, kwargs=None)


        if not use_numexpr:
            val = Subscript(value=Name(id='out', ctx=Load()), slice=index(i), ctx=Store())
            line = Assign(targets=[val], value=rhs )
        else:
            line = Expr(value=rhs) #Assign(targets=[Name(id='out_{}'.format(i), ctx=Load())], value=rhs )

        body.append(line)

        if output_names is not None:
            varname = symbols[output_names[0]][i]
            date = output_names[1]
            out_name = table_symbols[(varname,date)]
            line = Assign(targets=[Name(id=out_name.format(i), ctx=Store())],
                          value=Name(id='out_{}'.format(i), ctx=Store()))
            # body.append(line)


    args = [e[2] for e in arg_names] + ['out']

    # f = FunctionDef(name=funname, args=arguments(args=[Name(id=a, ctx=Param()) for a in args], vararg=None, kwarg=None, defaults=[]),
    #             body=preamble+body, decorator_list=[])

    f = FunctionDef(name=funname, args=arguments(args=[Name(id=a, ctx=Param()) for a in args], vararg=None, kwarg=None, kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=preamble+body, decorator_list=[])


    mod = Module(body=[f])
    mod = ast.fix_missing_locations(mod)

    print_code = True
    if print_code:

        s = "Function {}".format(mod.body[0].name)
        print("-"*len(s))
        print(s)
        print("-"*len(s))

        print( to_source(mod) )

    if return_ast:
        return mod
    else:
        fun = eval_ast(mod)
        return fun

def eval_ast(mod):

    from numexpr import evaluate

    context = {}

    context['division'] = division # THAT seems strange !

    import numpy

    context['inf'] = numpy.inf
    context['maximum'] = numpy.maximum
    context['minimum'] = numpy.minimum

    context['exp'] = numpy.exp
    context['log'] = numpy.log
    context['sin'] = numpy.sin
    context['cos'] = numpy.cos
    context['evaluate'] = evaluate

    context['abs'] = numpy.abs

    name = mod.body[0].name
    mod = ast.fix_missing_locations(mod)
    code  = compile(mod, '<string>', 'exec')
    exec(code, context, context)
    fun = context[name]
    return fun
