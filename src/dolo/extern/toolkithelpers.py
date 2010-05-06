# toolkithelpers:
# 1) qzswitch port
# 2) qzdiv port
# 3) the numpy-octave bridge class

from numpy import mat, c_, r_, multiply, where, sqrt, newaxis
from numpy.linalg import solve
from numpy.matlib import diag

def qzswitch(i, A2, B2, Q, Z):
    #print i, A2, B2, Q, Z
    Aout = A2.copy(); Bout = B2.copy(); Qout = Q.copy(); Zout = Z.copy()
    ix = i-1    # from 1-based to 0-based indexing...
    # use all 1x1-matrices for convenient conjugate-transpose even if real:
    a = mat(A2[ix, ix]); d = mat(B2[ix, ix]); b = mat(A2[ix, ix+1]);
    e = mat(B2[ix, ix+1]); c = mat(A2[ix+1, ix+1]); f = mat(B2[ix+1, ix+1])
    wz = c_[c*e - f*b, (c*d - f*a).H]
    xy = c_[(b*d - e*a).H, (c*d - f*a).H]
    n = sqrt(wz*wz.H)
    m = sqrt(xy*xy.H)
    if n[0,0] == 0: return (Aout, Bout, Qout, Zout)
    wz = solve(n, wz)
    xy = solve(m, xy)
    wz = r_[ wz, \
            c_[-wz[:,1].H, wz[:,0].H]]
    xy = r_[ xy, \
         c_[-xy[:,1].H, xy[:,0].H]]
    Aout[ix:ix+2, :] = xy * Aout[ix:ix+2, :]
    Bout[ix:ix+2, :] = xy * Bout[ix:ix+2, :]
    Aout[:, ix:ix+2] = Aout[:, ix:ix+2] * wz
    Bout[:, ix:ix+2] = Bout[:, ix:ix+2] * wz
    Zout[:, ix:ix+2] = Zout[:, ix:ix+2] * wz
    Qout[ix:ix+2, :] = xy * Qout[ix:ix+2, :]
    return (Aout, Bout, Qout, Zout)

def qzdiv(stake, A2, B2, Q, Z):
    Aout = A2.copy(); Bout = B2.copy(); Qout = Q.copy(); Zout = Z.copy()
    n, jnk = A2.shape
    # remember diag returns 1d
    root = mat(abs(c_[diag(A2)[:,newaxis], diag(B2)[:,newaxis]]))
    root[:,1] /= where(root[:,0]<1e-13, -root[:,1], root[:,0])
    for i in range(1,n+1)[::-1]:        # always first i rows, decreasing
        m = None
        for j in range(1,i+1)[::-1]:    # search backwards in the first i rows
            #print root.shape
            #print n, i, j
            #print 'i,j in qzdiv', i,j
            if root[j-1,1] > stake or root[j-1,1] < -0.1:
                m = j                   # get last relevant row
                break
        if m == None: return (Aout, Bout, Qout, Zout)
        for k in range(m,i):            # from relev. row to end of first part
            (Aout, Bout, Qout, Zout) = qzswitch(k, Aout, Bout, Qout, Zout)
            root[k-1:k+1, 1] = root[k-1:k+1, 1][::-1]
    return (Aout, Bout, Qout, Zout)

import numpy as n
from subprocess import Popen, PIPE
class octave4numpy:
    '''
    Lets octave functions calculate things, collects results as numpy-matrices.

    Example usage qz decomposition with octave syntax
     [AA,BB,Q,Z,V,W,lambda] = qz(A,B):

    1) "session approach"
    myoct = octave4numpy()
    myoct.definemats(['A', 'B'], [a, b])    # a and b: numpy arrays/matrices
    (AA,BB,Q,Z,V,W,lda) = myoct.execfunc('qz', ['A', 'B'], 7)
    # (7 is the number of returned objects, must be specified here in advance!)
    # (possibly execute other funcs after this)
    myoct.close()

    2) or the "shortcut approach", quick way for one-time use:
    myoct = octave4numpy('qz', (a, b), 7)
    (AA,BB,Q,Z,V,W,lda) = myoct.results
    # (connection to octave is automatically closed in this variant)

    If something else than 'octave' is needed to invoke your octave version,
      specify the command string in optional keyword arg cmd, like so:
    myoct = octave4numpy(cmd = 'octave2.0')

    The user should not pass 1d-numpy-arrays; be explicit about row or col!

    to do:
    - test on windows
    - allow to set precision
    - test if it also works with complex numbers (it should)
    ...
    '''
    def __init__(self, func = None, args = None, numout = 1, cmd = 'octave'):
        '''
        opens the octave connection via pipes (is this really cross-platform?)

        optionally already executes a function for quick use, func is the name
         of an octave function; then assignments must match the number of return
         objects from that specific octave function
        '''
        self.o = Popen([cmd, '-q'], stdin = PIPE, stdout = PIPE)  # -q: quiet
        if type(func) == type('f'):         # a function name was specified
            assert args != None
            if type(args) == type([]): args = tuple(args) # forgive user error
            elif type(args) != type((1,)):  # allow single matrix here
                args = (args,)              # convert to tuple
            # use generic names 'a0', 'a1', ...
            argnamelist = ['a' + str(ix) for ix in range(len(args))]
            self.definemats(argnamelist, args)
            self.results = self.execfunc(func, argnamelist, numout)
            self.octreturncode = self.close()

    def definemats(self, namelist, mlist):
        '''
        transfers matrices (values, can be 1x1) to octave for further use

        names must be valid denominator strings

        examples:
        myoct.definemats('a', m)
        myoct.definemats(['a', 'b'], [m1, m2])

        (since octave also accepts j for imaginary parts, should also work for
         complex numbers)
        '''
        if type(namelist) == type('n'):     # just one string instead of list
            namelist = [namelist]
            mlist = [mlist]
        assert len(namelist) == len(mlist)
        for name in namelist:
            m = mlist[namelist.index(name)]
            out =  ';'.join( \
                [','.join(map(str, row)) for row in n.mat(m).tolist()] )
            out =  '[' + out + ']'
            self.o.stdin.write(name + '=' + out + '\n')
            # just to clear stdout from noise:
            self.getreaction()

    def getreaction(self, items = 1):
        '''
        parse (and thereby chop off) octave's stdout

        Converting the string to a numpy matrix is left to another method.
        '''
        ## octave's pattern seems to be:
        ##   1) two '\n' after '=' (and thus before first matrix row)
        ##   2) one '\n' after each matrix row
        ##   3) two '\n' after last matrix row (and thus after each item)
        ## 
        ## After that it's an open-ended file and reading stdout stalls,
        ##  so we must avoid that!
        #
        # this is all very unpythonic, but if it works...
        output = ''
        itemcount = 0
        while itemcount < items:
            line = self.o.stdout.readline()
            output = output + line
            if output.endswith('\n\n') and output[-3] != '=': itemcount += 1
        #print output
        return output

    def octstr2nmat(self, octstring):
        '''
        creates numpy matrix from octave's matrix printout

        only one matrix per call should be passed
        '''
        # use only the second part, after the leading '...=' and w/o whitespace:
        temp = octstring.split('=')[1].strip()
        # convert row sep
        temp = temp.replace('\n', ';')
        ## deal with complex numbers;
        # octave uses 'i', python 'j'
        temp = temp.replace('i', 'j')
        # and octave's formatting 3 + 5i with blanks surrounding the plus sign
        #  produces numpy error, so strip blanks
        temp = temp.replace(' + ', '+')
        # (arbitrary number of blanks as col sep should be ok for n.mat)
        return n.mat(temp)

    def execfunc(self, funcname, argnames, numout = 1):
        '''
        executes an octave function call and returns matrix results

        For example, the octave function [AA,BB,Q,Z,V,W] = qz(A,B) would be
         called by
         (a, b, q, z, v, w) = myoctconn.exefunc('qz', ['myA', 'myB'], 6)

        The arg names must be given as strings, and must have been defined in
         octave before with definemats.

        If there's only one arg, it is admissible to provide just one arg string
         instead of an 1-element list
        '''
        if type(argnames) == type('a'): argnames = [argnames]
        # construct lhs, using generic names 'r0', 'r1', etc.
        lhs = '['
        for returnix in range(numout):
            lhs = lhs + 'r' + str(returnix) + ','
        lhs = lhs[:-1] + ']'        # stripping trailing comma
        # construct rhs
        rhs = funcname + '('
        for argname in argnames: rhs = rhs + argname + ','
        rhs = rhs[:-1] + ')'
        # execute
        self.o.stdin.write(lhs + '=' + rhs + '\n')
        outlist = []
        for item in range(numout):
            outlist.append(self.octstr2nmat(self.getreaction()))
        return outlist

    def close(self):
        self.o.communicate('quit')
        return self.o.returncode

#########
if __name__ == '__main__':
    print 'running testcode'
    # initial and debugging bits and pieces
    myo = octave4numpy()
    a = n.hstack( [n.ones((4,2)), n.random.rand(4,2)] )
    b = n.hstack( [n.zeros((4,2)), n.random.rand(4,2)] )
    myo.definemats('a', a)
    myo.definemats('b', b)
    # cleaning up
    print myo.close()
    del myo

    # some serious test cases
    print 'shortcut approach:'
    c = a
    d = b
    myo2 = octave4numpy('qz', (c, d), 7)
    # and then the result matrices are available as:
    for result in myo2.results: print result
    print 'octreturncode: ' + str(myo2.octreturncode)
    del myo2

    print 'session approach:'
    e = a
    f = b
    myoct = octave4numpy()
    myoct.definemats(['e', 'f'], [e, f])
    (AA,BB,Q,Z,V,W, lam) = myoct.execfunc('qz', ['e', 'f'], 7)
    myoct.close()
    print V
    print W