import time
from contextlib import contextmanager


@contextmanager
def timeit(msg):
    t1 = time.time()
    yield
    t2 = time.time()
    print("{}: {:.4f} s".format(msg, t2 - t1))
