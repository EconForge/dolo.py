import unittest


class Counter:
    n = 0

from dolo.misc.caching import clear_cache, cachedondisk, memoized, hashable

clear_cache()

@cachedondisk
def fun(a,b):
    Counter.n += 1
    return a + b

import dolo.config

class  CachingTestCase(unittest.TestCase):


    def test_cacheondisk(self):

        Counter.n = 0
        res1 = fun(1,2)
        assert( Counter.n == 1 )
        res2 = fun(1,2)
        assert( Counter.n == 1 )

    def test_cache_arrays(self):

        import numpy
        A = numpy.array([1,2,3,4]).reshape((2,2))
        B = numpy.array([0.1,0.2,0.3,0.4]).reshape((2,2))
        Counter.n = 0
        res1 = fun(A,B)
        assert( Counter.n == 1 )
        res2 = fun(A,B)
        assert( Counter.n == 1 )
        
if __name__ == '__main__':
    unittest.main()
