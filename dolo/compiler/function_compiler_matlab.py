import ast
import codegen

def str_to_expr(s):
    return ast.parse(s).body[0]

#class WriteMatlab(ast.NodeVisitor):
#
#    def visit_BinOp(self,b):
#        lhs = self.visit(b.left)
#        rhs = self.visit(b.right)
#        print( "{} + {}".format(lhs,rhs) ) """


def print_matlab(sexpr):
    from dolo.compiler.codegen import to_source
    ss = (to_source(sexpr))
    ss = ss.replace(' ** ', '.^')
    ss = ss.replace(' * ', '.*')
    ss = ss.replace(' / ', './')
    return ss



def compile_function_matlab(equations, symbols, arg_names, output_names=None, funname='anonymous'):

    from function_compiler_ast import std_date_symbol
    from function_compiler_ast import StandardizeDates

    from collections import OrderedDict
    table = OrderedDict()

    aa = arg_names
    # if output_names is not None:
        # aa = arg_names + [output_names]
    for a in aa:

        symbol_group = a[0]
        date = a[1]
        an = a[2]

        for b in symbols[symbol_group]:
            index = symbols[symbol_group].index(b)

            table[(b,date)] = (an, index)

    table_symbols = { k: (std_date_symbol(*k)) for k in table.keys() }

    code_preamble = ""
    for k,v in table.iteritems():
        std_name = table_symbols[k]
        if v[0] != 'p':
            code_preamble += "{} = {}(:,{});\n".format(std_name, v[0], v[1]+1)
        else:
            code_preamble += "{} = {}({});\n".format(std_name, v[0], v[1]+1)

    if output_names:
        out_group, out_time, out_s = output_names
    else:
        out_s = 'out'

    expressions = []

    code_expr = ""
    import sympy
    for i,eq in enumerate(equations):
        expr = str_to_expr(eq)
        sd = StandardizeDates(symbols, arg_names)
        sexpr = sd.visit(expr)
        expressions.append(sexpr)
        eq_string = (print_matlab(sexpr))
        if output_names is None:
            code_expr += "out(:,{}) = {} ;\n".format(i+1, eq_string)
        else:
            out_symbol = symbols[out_group][i]
            out_name = std_date_symbol(out_symbol, out_time)
            code_expr += "{} = {} ;\n".format(out_name, eq_string)
            code_expr += "{}(:,{}) = {} ;\n".format(out_s,i+1, out_name)


    ## temporary code to differentiate equations using sympy
    if output_names:
        # solve triangular system
        sympy_names = [sympy.Symbol(e) for e in symbols[out_group]]
        eq_dict = OrderedDict()
        for i,name in enumerate(sympy_names):
            eq = sympy.sympify(codegen.to_source(expressions[i]))
            eq_dict[name] = eq.subs(eq_dict)
        sympy_equations = eq_dict.values()
    else:
        sympy_equations = [sympy.sympify(codegen.to_source(expr)) for expr in expressions]


    all_derivatives = OrderedDict()
    for agn in arg_names:
        sym_group = agn[0]
        date = agn[1]
        short_sym_group =  agn[2]
        sym_names = [std_date_symbol(s, date) for s in symbols[sym_group]]
        dvals = []
        for eq in sympy_equations:
            deq = []
            for s in sym_names:
                sym = sympy.Symbol(s)
                ddif = (eq.diff(sym))
                deq.append(ast.parse(str(ddif)).body[0])
            dvals.append(deq)
        all_derivatives[short_sym_group] = dvals
    ## end of temporary code

    code_dexpr = ""
    for k,derivs in all_derivatives.iteritems():
        code_dexpr += '\n%derivatives w.r.t {}\n'.format(k)
        assign_name = "d_{}".format(k)
        code_dexpr += '{} = zeros(N,{},{});\n'.format(assign_name, len(derivs), len(derivs[0]))
        for i in range(len(derivs)):
            for j in range(len(derivs[0])):
                eq = derivs[i][j]
                ss = print_matlab(eq)
                if ss != '0':
                    code_dexpr += "{}(:,{},{}) = {};\n".format(assign_name, i+1, j+1, ss)

        code_dexpr += '\n'

    code = """\
function [{out_s}, {dout_s}] = {funname}({args_list})

%% preamble
{preamble}

N = size({first_arg},1);
out = zeros(N,{n_out});

%% equations

{equations}

%% derivatives

if nargout > 1
{dequations}
end

end
""".format(
    out_s = out_s,
    dout_s = str.join(', ', ["d_{}".format(e[2]) for e in arg_names]),
    preamble = code_preamble,
    funname = funname,
    args_list = str.join(', ', [e[2] for e in arg_names]),
    n_out = len(equations),
    first_arg = arg_names[0][2],
    equations = code_expr,
    dequations = code_dexpr
)

    return code

def compile_model_matlab(model):

    symbolic_model = model.symbolic

    model_spec = symbolic_model.model_spec

    if model_spec != 'fga':
        raise Exception("For now, only supported model type is fga")

    from collections import OrderedDict
    code_funs = OrderedDict()


    from dolo.compiler.recipes import recipes
    recipe = recipes['fga']
    symbols = model.symbols

    for funname, v in recipe['specs'].iteritems():

        spec = v['eqs']
        if 'complementarities' in v:
            print("ignoring complementarities")

        target = v.get('target')

        if funname not in symbolic_model.equations:
            continue
            # if not spec.get('optional'):
                # raise Exception("The model doesn't contain equations of type '{}'.".format(funname))
            # else:
                # continue

        eq_strings = symbolic_model.equations[funname]

        eq_strings = [eq.split('|')[0].strip() for eq in eq_strings]

        if target is not None:
            eq_strings = [eq.split('=')[1].strip() for eq in eq_strings]
        else:
            for i,eq in enumerate(eq_strings):
                if '=' in eq:
                    eq = '({1})-({0})'.format(*eq.split('=')[:2])
                eq_strings[i] = eq


        funcode = compile_function_matlab(eq_strings, symbols,
                        spec, output_names=target, funname=funname)

        code_funs[funname] = funcode


    code = """\
function [model] = construct_model()

    functions = struct;
"""
    for fn in code_funs.keys():
        code += "    functions.{} = @{};\n".format(fn, fn)

    code += "\n    symbols = struct;\n"
    for sg, sl in symbols.iteritems():
        code += "    symbols.{} = {{{}}};\n".format(sg, str.join(',', ["'{}'".format(e) for e in sl]))
    code += "\n"
    code += "\n    calibration = struct;"
    for sg, sl in model.calibration.iteritems():
        tv = [str(float(e)) for e in sl]
        tv = '[{}]'.format(str.join(', ', tv))
        code += "    calibration.{} = {};\n".format(sg, tv);

    # print covariances
    tv = [str(float(e)) for e in model.covariances.flatten()]
    tv = '[{}]'.format(str.join(', ', tv))
    n_c = model.covariances.shape[0]
    code += "\n    covariances = reshape( {} , {}, {});\n".format(tv, n_c, n_c)

    code += """\

    model = struct;
    model.functions = functions;
    model.calibration = calibration;
    model.symbols = symbols;
    model.covariances = covariances;

end

"""

    for fn, fc, in code_funs.iteritems():
        code += '\n'
        code += fc

    return code



if __name__ == '__main__':

    s1 = '(x0(1) + x1 / y0)**p0 - (x0(1) + x1 / y0)**(p0-1) '
    s2 = 'x0 + x1 / y1(+1)'

    expressions = [s1,s2]

    from collections import OrderedDict

    symbols = OrderedDict(
        states = ('x0', 'x1'),
        controls = ('y0','y1'),
        parameters = ('p0','p1')
    )

    arg_names = [
        ('states', 0, 's'),
        ('controls', 0, 'x'),
        ('states', 1, 'S'),
        ('controls', 1, 'X'),
        ('parameters', 0, 'p')
    ]

    import time

    t0 = time.time()
    resp = compile_function_matlab([s1,s2], symbols, arg_names) #funname='arbitrage', use_numexpr=True, return_ast=True)


    ###


    s1 = '(x0(1) + x1 / y0)**p0 - (x0(1) + x1 / y0)**(p0-1) '
    s2 = 'x0 + x1 / y1'

    expressions = [s1,s2]

    symbols = OrderedDict(
        states = ('x0', 'x1'),
        controls = ('y0','y1'),
        parameters = ('p0','p1')
    )

    arg_names = [
        ('states', 0, 's'),
        ('controls', 0, 'x'),
        ('states', 1, 'S'),
        ('parameters', 0, 'p')
    ]

    import time

    t0 = time.time()
    resp = compile_function_matlab([s1,s2], symbols, arg_names, output_names=('controls',1,'Y')) #funname='arbitrage', use_numexpr=True, return_ast=True)

    print(resp)

    print('***************#######################************************')

    from dolo import *
    model = yaml_import('examples/models/rbc.yaml')

    import time
    t1 = time.time()
    print( compile_model_matlab(model) )
    t2 = time.time()
