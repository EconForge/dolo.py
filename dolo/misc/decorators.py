import functools

from dolo.misc.caching import memoized, cachedondisk

import sys
is_python_3 =  sys.version_info >= (3, 0)

def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''

    import warnings

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        if is_python_3:
            code = func.__code__
        else:
            code = func.func_code
        warnings.warn_explicit(
            "Call to deprecated function {}.".format(func.__name__),
            category=Warning,
            filename=code.co_filename,
            lineno=code.co_firstlineno + 1
        )
        return func(*args, **kwargs)
    return new_func

@deprecated
def test_deprecation():
    pass
