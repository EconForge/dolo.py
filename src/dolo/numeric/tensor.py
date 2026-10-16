import numpy as np


def multidot_old(ten, mats):
    """
    Implements tensor operation : tensor-times-matrices.
    If last dimensions of ten represent multilinear operations of the type : [X1,...,Xk]->B[X1,...,Xk]
    and mats contains matrices or vectors [A1,...Ak] the function returns an array representing operators :
    [X1,...,Xk]->B[A1 X1,...,Ak Xk]
    """
    resp = ten
    n_d = ten.ndim
    n_m = len(mats)
    for i in range(n_m):
        # resp = np.tensordot( resp, mats[i], (n_d-n_m+i-1,0) )
        resp = np.tensordot(resp, mats[i], (n_d - n_m, 0))
    return resp


# should consist in shapes only
def mdot_signature(M, *C):
    M_syms = [chr(97 + e) for e in range(len(M))]
    fC_syms = M_syms[-len(C) :]
    ic = 97 + len(M_syms)
    C_syms = []
    for i in range(len(C)):
        c_sym = [fC_syms[i]]
        for j in range(len(C[i]) - 1):
            c_sym.append(chr(ic))
            ic += 1
        C_syms.append(c_sym)
    C_sig = [M_syms] + C_syms
    out_sig = [M_syms[: -len(C)]] + [cc[1:] for cc in C_syms]
    args = ",".join(["".join(g) for g in C_sig])
    out = "".join(["".join(g) for g in out_sig])

    return args + "->" + out


from numpy import einsum


def mdot(M, *C):
    sig = mdot_signature(M.shape, *[c.shape for c in C])
    return einsum(sig, M, *C)


def sdot(U, V):
    """
    Computes the tensorproduct reducing last dimensoin of U with first dimension of V.
    For matrices, it is equal to regular matrix product.
    """
    nu = U.ndim
    # nv = V.ndim
    return np.tensordot(U, V, axes=(nu - 1, 0))


def multitake(a, inds, axes):
    resp = a
    for n, ind in enumerate(inds):
        axe = axes[n]
        resp = resp.take(ind, axe)
    return resp
