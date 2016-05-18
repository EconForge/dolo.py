import numpy

def eval_with_diff(f, args, add_args, epsilon=1e-8):

    # f is a guvectorized function: f(x1, x2, ,xn, y1,..yp)
    # args is a list of vectors [x1,...,xn]
    # add_args is a list of vectors [y1,...,yn]
    # the function returns a list [r, dx1, ..., dxn] where:
    # r is the vector value value of f at (x1, xn, y1, yp)
    # dxi is jacobian w.r.t. xi

    # TODO: generalize when x1, ..., xn have non-core dimensions

    epsilon = 1e-8
    vec = numpy.concatenate(args)
    N = len(vec)
    points = vec[None,:].repeat(N+1, axis=0)
    for i in range(N):
        points[1+i,i] += epsilon

    argdims = [len(e) for e in args]
    cn = numpy.cumsum(argdims)
    slices = [e for e in zip( [0] + cn[:-1].tolist(), cn.tolist() )]
    vec_args = tuple([points[:,slice(*sl)] for sl in slices])

    arg_list = vec_args + add_args
    jac = f( *arg_list )
    res = jac[0,:]
    jac[1:,:] -= res[None,:]
    jac[1:,:] /= epsilon
    jacs = [jac[slice(sl[0]+1, sl[1]+1),:] for sl in slices]
    jacs = [j.T.copy() for j in jacs]  # to get C order
    return [res]  + jacs

class standard_function:

    epsilon = 1e-8

    def __init__(self, fun, n_output):

        # fun is a vectorized, non-allocating function
        self.fun = fun
        self.n_output = n_output

    def __call__(self, *args, diff=False, out=None):

        non_core_dims = [ a.shape[:-1] for a in args]
        core_dims = [a.shape[-1:] for a in args]

        non_core_ndims = [len(e) for e in non_core_dims]

        if (max(non_core_ndims) == 0):
            # we have only vectors, deal wwith it directly
            if not diff:
                if out is None:
                     out = numpy.zeros(self.n_output)
                self.fun(*(args+(out,)))
                return out

            else:
                def ff(*aa):
                    return self.__call__(*aa, diff=False)
                n_ignore = 1 # number of arguments that we don't differentiate
                res = eval_with_diff(ff, args[:-n_ignore], args[-n_ignore:], epsilon=1e-8)
                return res


        else:

            if not diff:
                K = max( non_core_ndims )
                ind = non_core_ndims.index( K )
                biggest_non_core_dim = non_core_dims[ind]
                biggest_non_core_dims = non_core_ndims[ind]
                new_args = []
                for i,arg in enumerate(args):
                    coredim = non_core_dims[i]
                    n_None = K-len(coredim)
                    n_Ellipsis = arg.ndim
                    newind = ((None,)*n_None) +(slice(None,None,None),)*n_Ellipsis
                    new_args.append(arg[newind])

                new_args = tuple(new_args)
                if out is None:
                    out = numpy.zeros( biggest_non_core_dim + (self.n_output,) )

                self.fun(*(new_args + (out,)))
                return out

            else:
                # older implementation
                return self.__vecdiff__(*args, diff=True, out=out)

    def __vecdiff__(self,*args, diff=False, out=None):


        fun = self.fun
        epsilon = self.epsilon

        sizes = [e.shape[0] for e in args if e.ndim==2]
        assert(len(set(sizes))==1)
        N = sizes[0]

        if out is None:
            out = numpy.zeros((N,self.n_output))

        fun( *( list(args) + [out] ) )

        if not diff:
            return out
        else:
            l_dout = []
            for i, a in enumerate(args[:-1]):
                # TODO: by default, we don't diffferentiate w.r.t. the last
                # argument. Reconsider.
                pargs = list(args)
                dout = numpy.zeros((N, self.n_output, a.shape[1]))
                for j in range( a.shape[1] ):
                    xx = a.copy()
                    xx[:,j] += epsilon
                    pargs[i] = xx
                    fun(*( list(pargs) + [dout[:,:,j]]))
                    dout[:,:,j] -= out
                    dout[:,:,j] /= epsilon
                l_dout.append(dout)
            return [out] + l_dout

################################

import ast
from dolo.compiler.symbolic import timeshift

class CountNames(ast.NodeVisitor):

    def __init__(self, known_variables, known_functions, known_constants):
        # known_variables: list of strings
        # known_functions: list of strings
        # known constants: list of strings

        self.known_variables = known_variables
        self.known_functions = known_functions
        self.known_constants = known_constants
        self.functions = set([])
        self.variables = set([])
        self.constants = set([])
        self.problems = []

    def visit_Call(self, call):
        name = call.func.id
        # colno = call.func.col_offset
        if name in self.known_variables:
            # try:
            assert(len(call.args) == 1)
            n = eval_scalar(call.args[0])
            self.variables.add((name, n))
            # except Exception as e:
            #     raise e
            #     self.problems.append([name, colno, 'timing_error'])
        elif name in self.known_functions:
            self.functions.add(name)
            for arg in call.args:
                self.visit(arg)
        elif name in self.known_constants:
            self.constants.add(name)
        else:
            self.problems.append(name)
            for arg in call.args:
                self.visit(arg)

    def visit_Name(self, cname):
        name = cname.id
        # colno = name.colno
        # colno = name.col_offset
        if name in self.known_variables:
            self.variables.add((name, 0))
        elif name in self.known_functions:
            self.functions.add(name)
        elif name in self.known_constants:
            self.constants.add(name)
        else:
            self.problems.append(name)


def parse(s): return ast.parse(s).body[0].value

# from http://stackoverflow.com/questions/1549509/remove-duplicates-in-a-list-while-keeping-its-order-python
def unique(seq):
    seen = set()
    for item in seq:
        if item not in seen:
            seen.add(item)
            yield item

def tshift(t, n):
    return (t[0], t[1]+n)


def get_deps(incidence, var, visited=None):

    # assert(var in incidence)
    assert(isinstance(var, tuple) and len(var) == 2)

    if visited is None:
        visited = (var,)
    elif var in visited:
        raise Exception("Non triangular system.")
    else:
        visited = visited + (var,)

    n = var[1]
    if abs(n) > 20:
        raise Exception("Found variable with time {}. Something has probably gone wrong.".format(n))

    deps = incidence[(var[0], 0)]
    if n != 0:
        deps = [tshift(e, n) for e in deps]

    resp = sum([get_deps(incidence, e, visited) for e in deps], [])

    resp.append(var)

    return resp

from ast import Name, Sub, Store, Assign, Subscript, Load, Index, Num, Call
from dolo.compiler.symbolic import eval_scalar, StandardizeDatesSimple, std_tsymbol
from collections import OrderedDict
from dolo.compiler.symbolic import match

def compile_function_ast(equations, symbols, arg_names, output_names=None, funname='anonymous', rhs_only=False,
            return_ast=False, print_code=False, definitions=None, vectorize=True, use_file=False):

    arguments = OrderedDict()
    for an in arg_names:
        if an[0] != 'parameters':
            t = an[1]
            arguments[an[2]] = [(s,t) for s in symbols[an[0]]]
    # arguments = [ [ (s,t) for s in symbols[sg]] for sg,t in arg_names if sg != 'parameters']
    parameters = [(s,0) for s in symbols['parameters']]
    targets = output_names
    if targets is not None:
        targets = [(s,targets[1]) for s in symbols[targets[0]]]

    mod = make_function(equations, arguments, parameters, definitions=definitions, targets=targets, rhs_only=rhs_only, funname=funname)

    from dolo.compiler.codegen import to_source
    import dolo.config
    if dolo.config.debug:
        print(to_source(mod))

    if vectorize:
        from numba import float64, void
        coredims = [len(symbols[an[0]]) for an in arg_names]
        signature = str.join(',', ['(n_{})'.format(d) for d in coredims])
        n_out = len(equations)
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

    from numba import jit, guvectorize

    jitted = jit(fun, nopython=True)
    if vectorize:
        gufun = guvectorize([fty], signature, target='parallel', nopython=True)(fun)
        return jitted, gufun
    else:
        return jitted
    return [f,None]


def make_function(equations, arguments, parameters, targets=None, rhs_only=False, definitions={}, funname='anonymous'):

    compat = lambda s: s.replace("^", "**").replace('==','=').replace('=','==')
    equations = [compat(eq) for eq in equations]

    if isinstance(arguments, list):
        arguments = OrderedDict( [('arg_{}'.format(i),k) for i, k in enumerate(arguments)])

    ## replace = by ==
    known_variables = [a[0] for a in sum(arguments.values(), [])]
    known_definitions = [a for a in definitions.keys()]
    known_parameters = [a[0] for a in parameters]
    all_variables = known_variables + known_definitions
    known_functions = []
    known_constants = []

    if targets is not None:
        all_variables.extend([o[0] for o in targets])
        targets = [std_tsymbol(o) for o in targets]
    else:
        targets = ['_out_{}'.format(n) for n in range(len(equations))]

    all_symbols = all_variables + known_parameters

    equations = [parse(eq) for eq in equations]
    definitions = {k: parse(v) for k, v in definitions.items()}

    defs_incidence = {}
    for sym, val in definitions.items():
        cn = CountNames(known_definitions, [], [])
        cn.visit(val)
        defs_incidence[(sym, 0)] = cn.variables
    # return defs_incidence
    from dolo.compiler.codegen import to_source
    equations_incidence = {}
    to_be_defined = set([])
    for i, eq in enumerate(equations):
        cn = CountNames(all_variables, known_functions, known_constants)
        cn.visit(eq)
        equations_incidence[i] = cn.variables
        to_be_defined = to_be_defined.union([a for a in cn.variables if a[0] in known_definitions])

    deps = []
    for tv in to_be_defined:
        ndeps = get_deps(defs_incidence, tv)
        deps.extend(ndeps)
    deps = [d for d in unique(deps)]

    sds = StandardizeDatesSimple(all_symbols)

    new_definitions = OrderedDict()
    for k in deps:
        val = definitions[k[0]]
        nval = timeshift(val, all_variables, k[1])  # function to print
        # dprint(val)
        new_definitions[std_tsymbol(k)] = sds.visit(nval)

    new_equations = []

    for n,eq in enumerate(equations):
        d = match(parse("_x == _y"), eq)
        if d is not False:
            lhs = d['_x']
            rhs = d['_y']
            if rhs_only:
                val = rhs
            else:
                val = ast.BinOp(left=rhs, op=Sub(), right=lhs)
        else:
            val = eq
        new_equations.append(sds.visit(val))



    # preambleIndex(Num(x))
    preamble = []
    for i,(arg_group_name,arg_group) in enumerate(arguments.items()):
        for pos,t in enumerate(arg_group):
            sym = std_tsymbol(t)
            rhs = Subscript(value=Name(id=arg_group_name, ctx=Load()), slice=Index(Num(pos)), ctx=Load())
            val = Assign(targets=[Name(id=sym, ctx=Store())], value=rhs)
            preamble.append(val)

    for pos,p in enumerate(parameters):
        sym = std_tsymbol(p)
        rhs = Subscript(value=Name(id='p', ctx=Load()), slice=Index(Num(pos)), ctx=Load())
        val = Assign(targets=[Name(id=sym, ctx=Store())], value=rhs)
        preamble.append(val)



    # now construct the function per se
    body = []
    for k,v in  new_definitions.items():
        line = Assign(targets=[Name(id=k, ctx=Store())], value=v)
        body.append(line)

    for n, neq in enumerate(new_equations):
        line = Assign(targets=[Name(id=targets[n], ctx=Store())], value=new_equations[n])
        body.append(line)

    for n, neq in enumerate(new_equations):
        line = Assign(targets=[Subscript(value=Name(id='out', ctx=Load()),
                                         slice=Index(Num(n)), ctx=Store())], value=Name(id=targets[n], ctx=Load()))
        body.append(line)


    from ast import arg, FunctionDef, Module
    from ast import arguments as ast_arguments

    from dolo.compiler.function_compiler_ast import to_source


    f = FunctionDef(name=funname,
                args=ast_arguments(args=[arg(arg=a) for a in arguments.keys()]+[arg(arg='p'),arg(arg='out')],
                vararg=None, kwarg=None, kwonlyargs=[], kw_defaults=[], defaults=[]),
                body=preamble + body, decorator_list=[])

    mod = Module(body=[f])
    mod = ast.fix_missing_locations(mod)
    return mod



def eval_ast(mod):

    context = {}


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
