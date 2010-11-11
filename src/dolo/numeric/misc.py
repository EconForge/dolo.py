
def iszero(mat):
    err = abs(mat).max()
    cond = err < 0.00000001
    if not cond:
        print 'Error : ' + str(err)
    return cond