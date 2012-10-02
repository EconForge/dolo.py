#from sympy import Symbol, symbols, exp, log
#
#[a,b,c,d] = symbols('a, b, c, d')
#
#[p,q] = symbols('p, q')
#
#
#arg1 = [a,b]
#arg2 = [c,d]
#args = [a,b,c,d]
#
#parms = [p,q]
#
#equations = [
#    log(a + b + exp(c*p)),
#    a * b**p - c**q*d*q
#]
#
#from dolo.compiler.compiling import compile_function_2
#
#txt = compile_function_2(equations, [arg1,arg2], ['x','y'], parms, return_function=False)
##print(txt)
#
#g = compile_function_2(equations, [arg1,arg2], ['x','y'], parms)
#
#from dolo.compiler.compiling_fast import compile_function_numexpr as compile_fast
#
#gg = compile_fast(equations, [arg1,arg2], ['x','y'], parms)
#
#
#print gg
#import numpy
#from numpy import zeros
#N = 10000
#x0 = numpy.row_stack([
#    zeros((1,N)) + 0.1,
#    zeros((1,N)) + 0.65
#])
#
#y0 = numpy.row_stack([
#    zeros((1,N)) + 1,
#    zeros((1,N)) + 3
#])
#
#p = numpy.array( [1.1, 2.2] )
#
#
#import time
#t0 = time.time()
#for i in range(100):
#    v1 = g(x0, y0, p, derivs=False)
#t1 = time.time()
#for i in range(100):
#    v2 = gg(x0, y0, p)
#t2 = time.time()
#
#print('Elapsed : ' + str(t1-t0) )
#print('Elapsed : ' + str(t2-t1) )
#
#print( abs(v2 - v1).max() )