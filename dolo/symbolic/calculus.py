from __future__ import division

import sympy

def second_order_taylorization(expr,shocks,covariances):    
    h = sympy.hessian(expr,shocks)
    resp = expr
    for i in range(covariances.shape[0]):
        for j in range(covariances.shape[1]):
            resp = resp + h[i,j] * covariances[i,j] / 2
    for s in shocks:
        resp = resp.subs(s,0)
    return(resp)


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
