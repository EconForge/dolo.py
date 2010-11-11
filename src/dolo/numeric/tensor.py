
def multidot(ten,mats):
    '''
    Implements tensor operation : tensor-times-matrices.
    If last dimensions of ten represent multilinear operations of the type : [X1,...,Xk]->B[X1,...,Xk]
    and mats contains matrices or vectors [A1,...Ak] the function returns an array representing operators : 
    [X1,...,Xk]->B[A1 X1,...,Ak Xk]
    '''
    resp = ten
    n_d = ten.ndim
    n_m = len(mats)
    for i in range(n_m):
        #resp = np.tensordot( resp, mats[i], (n_d-n_m+i-1,0) )
        resp = np.tensordot( resp, mats[i], (n_d-n_m,0) )
    return resp
