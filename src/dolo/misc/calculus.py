# -*- coding: utf-8 -*-
import sympy
#from scipy import *
from dolo.model.symbolic import *
#import dolo.extern.qz
import copy

#def second_order_taylorisation(expr,shocks,covariances):
def second_order_taylorization(expr,shocks,covariances):    
    h = sympy.hessian(expr,shocks)
    resp = expr
    for i in range(covariances.shape[0]):
        for j in range(covariances.shape[1]):
            resp = resp + h[i,j] * covariances[i,j] / 2
    for s in shocks:
        resp = resp.subs(s,0)
    return(resp)
    
    
def lambda_sub(expr,dic):
    # This function evaluates expression expr to a float using float dictionary dic
    # dic must contains at least all symbols of expr
    expr = sympy.core.sympify(expr)
    symbs = [s for s in expr.atoms() if isinstance(s,sympy.Symbol)]
    vals = [ dic[s] for s in symbs]
    f = sympy.lambdify(symbs,expr)
    res = (f(*vals))
    return(res)
    

def solve_triangular_system(sdict,return_order=False,unknown_type=sympy.Symbol):
    oks = []
    symbols = dict()
    for v in sdict:
        atoms = sympy.core.sympify(sdict[v]).atoms()
        symbols[v] = set([s for s in atoms if isinstance(s,unknown_type)])
    notfinished = True
    while notfinished:
        l = len(symbols)
        symkeys = symbols.keys()
        for s in symkeys:
            if len(symbols[s]) == 0:
                oks.append(s)
                symbols.pop(s)
                for ss in symbols:
                    symbols[ss].discard(s)
        if len(symbols) == 0:
            notfinished = False
        elif len(symbols) == l:
            raise(Exception('The system is not triangular'))
    if return_order:
        return(oks)
    else:
        res = copy.copy(sdict)
        for s in oks:
            res[s] = lambda_sub(res[s],res)
        return [res,oks]

def construct_4_blocks_matrix(blocks):
    '''construct block matrix line by line
    input : [[A1,A2],[A3,A4]] 
    '''
    A1 = blocks[0][0]
    A2 = blocks[0][1]
    A3 = blocks[1][0]
    A4 = blocks[1][1]

    [p1,q1] = (blocks[0][0]).shape
    [p2,q2] = (blocks[0][1]).shape
    [p3,q3] = (blocks[1][0]).shape
    [p4,q4] = (blocks[1][1]).shape 
    if p1<>p2 or p3<>p4 or q1<>q3 or q2<>q4:
        raise('dimension mismatch')
    m = zeros((p1+p3,q1+q2))

    m[0:p1,0:q1] = A1
    m[0:p1,q1:(q1+q2)] = A2
    m[p1:p1+p3,0:q1] = A3
    m[p1:p1+p3,q1:(q1+q2)] = A4

    return(m)
    
def sympy_to_dynare_string(sexpr):
    s = str(sexpr)
    s = s.replace("==","=")
    s = s.replace("**","^")
    return(s)
