import numpy as np
import ctypes
f  = ctypes.cdll.LoadLibrary('/home/pablo/Programmation/bigeco/dolo/src/dolo/numeric/c/serial_operations_lib.so')
fun = f['cserop']

def serial_multiplication(A,B):

    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B)
    
    [I,J,N] = A.shape
    K = B.shape[1]
    Z = np.zeros( (I,K,N) )

    A_ptr = A.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    B_ptr = B.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    Z_ptr = Z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    fun(I,J,K,N,A_ptr,B_ptr,Z_ptr)

    return Z
