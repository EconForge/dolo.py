import numpy.random

from dolo.numeric.serial_operations import strange_tensor_multiplication
from dolo.numeric.newton import serial_inversion

import time

I = 100
J = 10
K = 10
N = 2000

A = numpy.random.random( (I,J,N) )
B = numpy.random.random( (J,K,N) )


t = time.time()
for i in range(10):
    C = strange_tensor_multiplication(A,B)
s = time.time()
for i in range(10):
    CC = strange_tensor_multiplication(A,B)
u = time.time()


print('Elapsed : ' +str(s-t) )


print('Elapsed : ' +str(u-s) )


# 4 times faster