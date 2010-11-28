'''
QZ alias generalized Schur decomposition (complex or real) for Python/Numpy.

You need to import the qz() function of this module, check out its docstring,
especially what it says about the required lapack shared library. Run this 
module for some quick tests of the setup.
 
This is free but copyrighted software, distributed under the same license
as Python 2.5, copyright Sven Schreiber.

If you think a different license would make (more) sense, please say so
on the Numpy mailing list (see scipy.org).
''' 

from ctypes import cdll, c_int, c_char, POINTER
import numpy as np
from numpy.ctypeslib import load_library, ndpointer
import sys
import os

import sys
sys.stderr = open("logfile.txt","w")

__dirname__ =  os.path.dirname(__file__)

__lapack_location__ = __dirname__
__lapack_name__ = 'lapack.dll'
__libio_name__ = 'libiomp5md.dll' 

def setuplapack4xgges(A,B,lpname,lppath):
    '''Loads the lapack shared lib and does some input checks.
    
    The defaults for lapackname and location are platform-specific:
        Win32: 'lapack' (due to scilab's lapack.dll)
               'c:\\winnt\\system32\\'
        Otherwise: 'liblapack' 
                   '/usr/lib/'
    '''
    # some input checks
    assert A.ndim == 2
    assert A.shape == B.shape
    assert A.shape[0] == A.shape[1]

    
    print  __lapack_location__+'\\'+__libio_name__
    print __lapack_location__+'\\'+__lapack_name__
    try:
        __libio__ = cdll.LoadLibrary( __lapack_location__+'\\'+__libio_name__)
        lapack = cdll.LoadLibrary( __lapack_location__+'\\'+__lapack_name__)
    except Exception as e:
        raise ImportError('lapack')

    return lapack

def dgges4numpy(A,B, jobvsl='V', jobvsr='V', lapackname='', lapackpath=''):
    '''wraps lapack function dgges, no sorting done'''
    lapack = setuplapack4xgges(A,B,lapackname,lapackpath)
    rows = A.shape[0]
    # to determine matrix subclass
    Aintype = type(A)
    
    # actual inputs
    A = np.asfortranarray(A, dtype=np.float64)
    B = np.asfortranarray(B, dtype=np.float64)
    # seems ok to pass strings directly, but the function expects only 1 char!
    jobvsl = jobvsl[0]
    jobvsr = jobvsr[0]
    
    # dummy inputs
    sort    = 'N'            # we don't want sorting
    dummy   = 0     # 
    info    = c_int(1)
    lda     = c_int(rows)
    ldb     = c_int(rows)
    ldvsl   = c_int(rows)
    ldvsr   = c_int(rows)
    plwork  = 16*rows        # needed again later
    lwork   = c_int(lwork)
    n       = c_int(rows)
    csdim   = c_int(rows)    # because we don't sort
    
    # auxiliary arrays
    Alphar = np.asfortranarray(np.empty(rows), dtype=np.float64)
    Alphai = np.asfortranarray(np.empty(rows), dtype=np.float64)
    Beta  = np.asfortranarray(np.empty(rows), dtype=np.float64)
    Vsl   = np.asfortranarray(np.empty([rows,rows]), dtype=np.float64)
    Vsr   = np.asfortranarray(np.empty([rows,rows]), dtype=np.float64)
    Work  = np.asfortranarray(np.empty(plwork), dtype=np.float64)
    Rwork = np.asfortranarray(np.empty(8*rows), dtype=np.float64)
    
    lapack.dgges_.argtypes = [  
        POINTER(c_char),                                       # JOBVSL
        POINTER(c_char),                                       # JOBVSR
        POINTER(c_char),                                       # SORT
        # for the dummy the POINTER thing didn't work, 
        #  but plain c_int apparently does...
        c_int,                                          # dummy SELCTG 
        POINTER(c_int),                                        # N
        ndpointer(dtype=np.float64, ndim=2, flags='FORTRAN'),  # A
        POINTER(c_int),                                        # LDA
        ndpointer(dtype=np.float64, ndim=2, flags='FORTRAN'),  # B
        POINTER(c_int),                                        # LDB
        POINTER(c_int),                                        # SDIM
        ndpointer(dtype=np.float64, ndim=1, flags='FORTRAN'),  # ALPHAr
        ndpointer(dtype=np.float64, ndim=1, flags='FORTRAN'),  # ALPHAi
        ndpointer(dtype=np.float64, ndim=1, flags='FORTRAN'),  # BETA
        ndpointer(dtype=np.float64, ndim=2, flags='FORTRAN'),  # VSL
        POINTER(c_int),                                        # LDVSL
        ndpointer(dtype=np.float64, ndim=2, flags='FORTRAN'),  # VSR
        POINTER(c_int),                                        # LDVSR
        ndpointer(dtype=np.float64, ndim=1, flags='FORTRAN'),  # WORK
        POINTER(c_int),                                        # LWORK
        # same as with SELCTG...
        c_int,                                                 # dummy BWORK 
        POINTER(c_int) ]                                       # INFO
    
    lapack.dgges_(jobvsl,jobvsr,sort,dummy,n,A,lda,B,ldb,sdim,Alphar,Alphai,
                Beta,Vsl,ldvsl,Vsr,ldvsr,Work,lwork,dummy,info)
    
    # preserve matrix subclass
    if Aintype == type(np.mat(1)):
        A=np.mat(A); B=np.mat(B); Vsl=np.mat(Vsl); Vsr=np.mat(Vsr)
    if info.value == 0:
        if   jobvsl=='V' and jobvsr=='V': return A,B,Alphar,Alphai,Beta,Vsl,Vsr
        elif jobvsl=='V' and jobvsr=='N': return A,B,Alphar,Alphai,Beta,Vsl
        elif jobvsl=='N' and jobvsr=='V': return A,B,Alphar,Alphai,Beta,Vsr
        else:                             return A,B,Alphar,Alphai,Beta
    elif info.value < 0:
        raise ValueError, 'Illegal argument (' + str(abs(info.value)) + ')'
    elif info.value <= rows: 
        raise RuntimeError, 'QZ iteration failed'
    elif info.value <= rows+3:
        raise RuntimeError, 'something other than QZ iteration failed'
    else: raise RuntimeError, 'INFO not updated by dgges, complete failure!?'

def zgges4numpy(A,B, jobvsl='V', jobvsr='V', lapackname='', lapackpath=''):
    '''Wraps lapack function zgges, no sorting done.
    
    Returns complex arrays, use real_if_close() if needed/possible.
    '''
    lapack = setuplapack4xgges(A,B,lapackname,lapackpath)
    rows = A.shape[0]
    # determine matrix subclass
    Aintype = type(A)
        
    # actual inputs
    # The COMPLEX*16 type in Fortran translates to numpy's complex128
    A = np.asfortranarray(A, dtype=np.complex128)
    B = np.asfortranarray(B, dtype=np.complex128)
    # seems ok to pass strings directly, but the function expects only 1 char!
    jobvsl = jobvsl[0]
    jobvsr = jobvsr[0]
    
    # dummy inputs
    sort = 'N'         # we don't want sorting
    dummy = 0           # a placeholder for what would be needed for sorting 
    info = c_int(rows+4)  # >n+3 aren't used as error codes of zgges
    lda = c_int(rows)
    ldb = c_int(rows)
    ldvsl = c_int(rows)
    ldvsr = c_int(rows)
    plwork = 16*rows        # needed again later
    lwork = c_int(plwork)
    n = c_int(rows)
    sdim = c_int(0)    # because we don't sort
    
    # auxiliary arrays
    Alpha = np.asfortranarray(np.empty(rows), dtype=np.complex128)
    Beta  = np.asfortranarray(np.empty(rows), dtype=np.complex128)
    Vsl   = np.asfortranarray(np.empty([rows,rows]), dtype=np.complex128)
    Vsr   = np.asfortranarray(np.empty([rows,rows]), dtype=np.complex128)
    Work  = np.asfortranarray(np.empty(plwork), dtype=np.complex128)
    Rwork = np.asfortranarray(np.empty(8*rows), dtype=np.float64)
    
    lapack.zgges_.argtypes = [  
        POINTER(c_char),                                         # JOBVSL
        POINTER(c_char),                                         # JOBVSR
        POINTER(c_char),                                         # SORT
        c_int,                                             # dummy SELCTG 
        POINTER(c_int),                                          # N
        ndpointer(dtype=np.complex128, ndim=2, flags='FORTRAN'), # A
        POINTER(c_int),                                          # LDA
        ndpointer(dtype=np.complex128, ndim=2, flags='FORTRAN'), # B
        POINTER(c_int),                                          # LDB
        POINTER(c_int),                                          # SDIM
        ndpointer(dtype=np.complex128, ndim=1, flags='FORTRAN'), # ALPHA
        ndpointer(dtype=np.complex128, ndim=1, flags='FORTRAN'), # BETA
        ndpointer(dtype=np.complex128, ndim=2, flags='FORTRAN'), # VSL
        POINTER(c_int),                                          # LDVSL
        ndpointer(dtype=np.complex128, ndim=2, flags='FORTRAN'), # VSR
        POINTER(c_int),                                          # LDVSR
        ndpointer(dtype=np.complex128, ndim=1, flags='FORTRAN'), # WORK
        POINTER(c_int),                                          # LWORK
        ndpointer(dtype=np.float64, ndim=1, flags='FORTRAN'),    # RWORK
        c_int,                                             # dummy BWORK 
        POINTER(c_int) ]                                         # INFO
    
    lapack.zgges_(jobvsl,jobvsr,sort,dummy,n,A,lda,B,ldb,sdim,Alpha,
                 Beta,Vsl,ldvsl,Vsr,ldvsr,Work,lwork,Rwork,dummy,info)
    
    # preserve matrix subclass
    if Aintype == type(np.mat(1)):
        A=np.mat(A); B=np.mat(B); Vsl=np.mat(Vsl); Vsr=np.mat(Vsr)
    # use .value for ctypes safety, although probably redundant
    if info.value == 0:
        if   jobvsl=='V' and jobvsr=='V': return A,B,Alpha,Beta,Vsl,Vsr
        elif jobvsl=='V' and jobvsr=='N': return A,B,Alpha,Beta,Vsl
        elif jobvsl=='N' and jobvsr=='V': return A,B,Alpha,Beta,Vsr
        else:                             return A,B,Alpha,Beta
    elif info.value < 0:
        raise ValueError, 'Illegal argument (' + str(abs(info.value)) + ')'
    elif info.value <= rows: 
        raise RuntimeError, 'QZ iteration failed'
    elif info.value <= rows+3:
        raise RuntimeError, 'something other than QZ iteration failed'
    else: raise RuntimeError, 'INFO not updated by zgges, complete failure!?'

def qz(A,B, mode='complex', lapackname='', lapackpath=''):
    '''Equivalent to Matlab's qz function [AA,BB,Q,Z] = qz(A,B).
    
    Requires Lapack as a shared compiled library on the system (one that
    contains the functions dgges for real and zgges for complex use -- on 
    Windows the one shipped with Scilab works). The underlying defaults for 
    lapackname and lapackpath are platform-specific:
        Win32: 'lapack' (due to scilab's lapack.dll)
               'c:\\winnt\\system32\\'
        Otherwise: 'liblapack' 
                   '/usr/lib/'
    
    This function should exactly match Matlab's usage, unlike octave's qz 
    function which returns the conjugate-transpose of one of the matrices. Thus
    it holds that 
        AA = Q*A*Z
        BB = Q*B*Z,
    where Q and Z are unitary (orthogonal if real).
     
    If mode is 'complex', then:
     returns complex-type arrays, 
     AA and BB are upper triangular, 
     and diag(AA)/diag(BB) are the generalized eigenvalues of (A,B).
     
    If the real qz decomposition is explicitly requested --as in Matlab:  
    qz(A,B,'real')-- then:
     returns real-type arrays,
     AA is only block upper triangular,
     and calculating the eigenvalues is more complicated.
     
    Other variants like [AA,BB,Q,Z,V,W] = qz(A,B) are not implemented, i.e.
    no generalized eigenvectors are calculated.
    '''
    if mode == 'real':
        AA,BB,dum1,dum2,dum3,VSL,VSR = dgges4numpy(A,B,
                        lapackname=lapackname,lapackpath=lapackpath)
        return AA, BB, VSL.T, VSR
    elif mode == 'complex':
        AA,BB,dum1,dum2,VSL,VSR = zgges4numpy(A,B,
                        lapackname=lapackname,lapackpath=lapackpath)
        return AA, BB, VSL.conj().T, VSR
    else: raise ValueError, 'bogus choice for mode'
   
def eig2(A,B, lapackname='', lapackpath=''):
    '''Calculates generalized eigenvalues of pair (A,B).
    
    This should correspond to Matlab's lambda = eig(A,B),
    and also to some (the same?) scipy function.
    
    Eigenvalues will be of complex type, are unsorted, and are returned as 1d.
    '''
    AA,BB,dum1,dum2,VSL,VSR = zgges4numpy(A,B,
                    lapackname=lapackname,lapackpath=lapackpath)
    return np.diag(AA)/np.diag(BB)
    
def eigwithqz(A,B, lapackname='', lapackpath=''):
    '''Does complex QZ decomp. and also returns the eigenvalues'''
    AA, BB, Q, Z = qz(A,B,lapackname=lapackname,lapackpath=lapackpath)
    evals = np.diag(AA)/np.diag(BB)
    return evals,AA,BB,Q,Z
    
if __name__ == '__main__': 
    #print help(np.asfortranarray)
    
    # test case:
    myA = np.random.rand(4,4)
    myB = np.random.rand(4,4)
    myqz = qz(myA, myB, lapackname='lapack')
    AAd = np.diag(myqz[0])
    BBd = np.diag(myqz[1])
    myeig = eig2(myA,myB,lapackname='lapack')
    assert np.allclose(AAd/BBd, myeig)
    print myeig
    print eigwithqz(myA,myB)
    
    # does it preserve matrices?
    print type(qz(np.mat(myA),np.mat(myB))[0])
    assert type(qz(np.mat(myA),myB)[0]) == type(np.mat(np.eye(2)))
