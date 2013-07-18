from __future__ import division

from dolo.symbolic.derivatives import DerivativesTree
from dolo.compiler.common import DicPrinter
from dolo.symbolic.symbolic import TSymbol

DerivativesTree.symbol_type = TSymbol

from numbapro import float64
#from numbapro.vectorize import guvectorize

def compile_multiargument_function(equations, args_list, args_names, parms, fname='anonymous_function', diff=True, return_text=False, order='columns'):

    """
    :param equations: list of sympy expressions
    :param args_list: list of lists of symbols (e.g. [[a_1,a_2], [b_1,b_2]])
    :param args_names: list of strings (['a','b']
    :param parms: list of symbols to be used as parameters
    :param fname: name of the python function to be generated
    :param diff: include symbolic derivates in generated function
    :param vectorize: arguments are vectorized (not parameters)
    :param return_function: a python function f(a,b,p) where p is a vector of parameters and a, b, arrays
    :return:
    """

    if order == 'rows':
        raise(Exception('not implemented'))

    template = '{0}[i,{1}]'

    declarations = ""
    sub_list = {}

    for i,args in enumerate(args_list):
        vec_name = args_names[i]
        for j,v in enumerate(args):
            sub_list[v] = template.format(vec_name,j)
            # declarations += "    {0}_{1} = {0}[{1}]\n".format(vec_name, j)

    for i,p in enumerate(parms):
        sub_list[p] = '{0}[{1}]'.format('p',i)
        # declarations += "    {0}_{1} = {0}[{1}]\n".format('p', i)

    import sympy
    # TODO: construct a common list of symbol that should be understood everywhere
    sub_list[sympy.Symbol('inf')] = 'inf'

    args_size = [len(e) for e in args_list] + [len(parms)]
    return_size = len(equations)

    size_same_as_output = args_size.index(return_size)

    signature = str.join(',',['(n{})'.format(i) for i in range(len(args_size))])
    signature += '->(n{})'.format(return_size-1)
    argtypes = [float64[:,:]]*(len(args_size)-1) + [float64[:], float64[:,:]] 


    text = '''
from numbapro import float64, void
from numbapro import jit
from numbapro import cuda

@jit( argtypes={argtypes}, target='gpu' )
def {fname}({args_names}, {param_names}, val):

    i = cuda.grid(1)
{content}

#    return val



'''

    from dolo.compiler.common import DicPrinter

    dp = DicPrinter(sub_list)

    def write_eqs(eq_l,outname='val'):
        eq_block = ''
        for i,eq in enumerate(eq_l):
            eq_string = dp.doprint_numpy(eq)
            eq_block += '    val[i,{0}] = {1}\n'.format(i, eq_string)
        return eq_block

    content = write_eqs(equations)



    return_names = 'val'
    text = text.format(
            fname = fname,
            n_equations = len(equations),
            declarations = declarations,
            var = args_names[0],
            content = content,
            return_names = return_names,
            args_names = str.join(', ', args_names),
            param_names = 'p',
            argtypes = str(argtypes),
            signature = signature,
    )


    if return_text:
        return text
    print(args_list)

    return code_to_function(text,fname,args_size,return_size, size_same_as_output)


def code_to_function(text, name, args_size, return_size, size_same_as_output):
    from numpy import zeros
    from numpy import exp, log
    from numpy import sin, cos, tan
    from numpy import arcsin as asin
    from numpy import arccos as acos
    from numpy import arctan as atan
    from numpy import sinh, cosh, tanh
    from numpy import pi
    from numpy import inf
    from numbapro import cuda
    d = locals()
    e = {}
    print('*'*1000)
    print(text)
    exec(text, d, e)
    fun = e[name]
    fun.max_blocksize = 16
    return fun


def eqdiff(leq,lvars):
    resp = []
    for eq in leq:
        el = [ eq.diff(v) for v in lvars]
        resp += [el]
    return resp



if __name__ == '__main__':

    from dolo import *
    import numpy


    import yaml
    with file('../washington/code/recipes.yaml') as f:
        recipes = yaml.load(f)

    fname = '..//washington/code/rbc_fg.yaml'

    first = 'numexpr'
    second = 'numba_gpu'

    gm = yaml_import(fname, compiler=first, order='columns', recipes=recipes)
    gmp = yaml_import(fname, compiler=second, order='columns', recipes=recipes)


    ss = gmp.calibration['states']
    xx = gmp.calibration['controls']
    p = gmp.calibration['parameters']
    ee = numpy.array([0],dtype=numpy.double)

    N = 32*1000

    ss = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ss), (N,1) ) )
    xx = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(xx), (N,1) ) )
    ee = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(ee), (N,1) ) )
    pp = numpy.ascontiguousarray( numpy.tile(numpy.atleast_2d(p), (N,1) ) )



    g = gm.functions['transition']
    f = gm.functions['arbitrage']

    gp = gmp.functions['transition']
    fp = gmp.functions['arbitrage']

    import time

    from numbapro.cuda import to_device

    g_ss = to_device(ss)
    g_xx = to_device(xx)
    g_p = to_device(p[None,:])

#    exit()

    res_test = f(ss,xx,ss,xx,p)
    
    res_gpu = fp(g_ss,g_xx,g_ss,g_xx,g_p)
   
    res = res_gpu.copy_to_host()

    print('Error : {}'.format( abs(res - res_test).max() ) )










    exit()
    print(first)

    tmp = g(ss,xx,ee,p)
    t1 = time.time()
    for i in range(50):
        tmp = g(ss,xx,ee,p)
    t2 = time.time()

    tmp = f(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp = f(ss,xx,ss,xx,p)
    t4 = time.time()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))

    from numbapro.cuda import to_device

    g_ss = to_device(ss)
    g_xx = to_device(xx)
    g_ee = to_device(ee)
    g_p = to_device(p)
    g_o = to_device(ss.copy()) 
    print(second)
    print(g_ss.shape)
    print(g_xx.shape)
    print(g_ee.shape)
    print(g_p.shape)
    g_o = gp(g_ss,g_xx,g_ee,g_p)
    print('Success !')
    t1 = time.time()
    for i in range(50):
        print(i)
        g_o = gp(g_ss,g_xx,g_ee,g_p)
    t2 = time.time()
#    exit()
    tmp2 = fp(ss,xx,ss,xx,p)
    t3 = time.time()
    for i in range(50):
        tmp2 = fp(ss,xx,ss,xx,p)
    t4 = time.time()

    tmp2 = tmp2.copy_to_host()

    print('first {}'.format(t2-t1))
    print('second {}'.format(t4-t3))

    print('Error : {}'.format(abs(tmp2-tmp).max()) )
