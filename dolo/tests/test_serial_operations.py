#import pyximport; pyximport.install()
#
#
#import numpy.random
#
#from dolo.numeric.serial_operations import serial_multiplication, serial_dot, strange_tensor_multiplication
#from dolo.numeric.serial_operations_c import serial_multiplication as smult_c
#from dolo.numeric.serial_operations_cython import serial_multiplication as smult_cython
#
#
#
#
#from dolo.numeric.serial_operations import serial_inversion
#from dolo.numeric.serial_operations_cython import serial_inversion as serial_inversion_cython
#
#
##from dolo.numeric.serial_operations_cython import strange_tensor_multiplication as stm
##from dolo.numeric.serial_operations_cython import scratch_tm as sstm
##from dolo.numeric.serial_operations_cython import serial_dot as sdot
##from dolo.numeric.serial_operations_cython import serop_from_c
#
#
#
#import time
#
#I = 10
#J = 10
#K = 10
#N = 2000
#
#A = numpy.random.random( (I,J,N) )
#B = numpy.random.random( (J,K,N) )
#
##A = numpy.asfortranarray(A)
##B = numpy.asfortranarray(B)
#
#n_exp = 10
#
#
#
#
#t = time.time()
#for i in range(n_exp):
#    C = strange_tensor_multiplication(A,B)
#s = time.time()
#print('S.T.M. (python) ' + str(s-t))
#
#
#n_exp = 10
#t = time.time()
#for i in range(n_exp):
#    CC = serial_multiplication(A,B)
#s = time.time()
#print('S.T.M. (c) ' + str(s-t))
#
#n_exp = 10
#t = time.time()
#for i in range(n_exp):
#    CCC = smult_cython(A,B)
#s = time.time()
#print('S.T.M. (cython) ' + str(s-t))
#
#print abs(CC - C).max()
#print abs(CCC - C).max()
#
#
#exit()
#
#
#t = time.time()
#for i in range(n_exp):
#    CCC = serial_inversion(A)
#s = time.time()
#print('S.I. (python) ' + str(s-t))
#
#t = time.time()
#for i in range(n_exp):
#    CCC = serial_inversion_cython(A)
#s = time.time()
#print('S.I. (python) ' + str(s-t))
#
#exit()
#
#exit()
#t = time.time()
#for i in range(n_exp):
#    CC = another_multiplication(A,B)
#s = time.time()
#print('S.T.M. (c) ' + str(s-t))
#
#
#
#exit()
#
#
#exit()
#
#
#
#
#print( abs(CC - C).max())
#
#
#t = time.time()
#for i in range(10):
#    CCC = serial_dot(A,B)
#s = time.time()
#print('S.D. (python) ' + str(s-t))
#
#exit()
#
#
#import numpy as np
#AA = (A)
#BB = (B)
#t = time.time()
#for i in range(10):
#    CCCC = sdot(AA,BB)
#s = time.time()
#print('S.D. (cython) ' + str(s-t))
#
## 4 times faster
#print C.shape
#print CCCC.shape
#print( abs(CCCC - C).max() )