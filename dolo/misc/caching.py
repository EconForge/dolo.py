import functools


class memoized(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args, **kargs):

        targs = (e for e in args)
        hh = tuple(  hashable(e) for e in targs )
        h2 = hashable(kargs)
        h = hash( (hh, h2) )
        try:
            return self.cache[h]
        except KeyError:
            value = self.func(*args, **kargs)
            self.cache[h] = value
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args, **kargs)
    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__
    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)


class cachedondisk(object):
    """Decorator that caches a function's return value each time it is called.
    If called later with the same arguments, the cached value is returned, and
    not re-evaluated.
    """

    def __init__(self, func):

        # create caching direcory if it is not there already
        import os
        if not os.path.isdir('.cache'):
            os.mkdir('.cache')

        self.func = func
        self.fname = func.__name__

    def __call__(self, *args, **kargs):
        import pickle
        hh = tuple(  hashable(e) for e in args )
        h2 = hashable(kargs)
        h = hash( (hh, h2) )
        try:
            with open('.cache/{0}.{1}.pickle'.format(self.fname,h),'rb') as f:
                value = pickle.load(f)
            return value
        except IOError:
            value = self.func(*args, **kargs)
            if value is not None:  # should there be other kinds of error values
                # write file with h
                with open('.cache/{0}.{1}.pickle'.format(self.fname,h),'wb') as f:
                    pickle.dump(value,f)
            return value
        except TypeError:
            # uncachable -- for instance, passing a list as an argument.
            # Better to not cache than to blow up entirely.
            return self.func(*args, **kargs)
    def __repr__(self):
        """Return the function's docstring."""
        return self.func.__doc__
    def __get__(self, obj, objtype):
        """Support instance methods."""
        return functools.partial(self.__call__, obj)

def clear_cache(function_name=None):

    import os

    try:
        if function_name:
            os.system('rm -rf .cache/{}.*.pickle'.format(function_name))
        else:
            os.system('rm -rf .cache/*.pickle')
    except:
        pass

import os

class DiskDictionary:

    def __init__(self,directory='.cache',funname='fun'):
        self.funname = funname
        if not os.path.isdir(directory):
            os.mkdir(directory)
        self.directory = directory

    def get_filename(self, key):
        import pickle
        hh = tuple(hashable(k) for k in key)
        h = hash(hh)
        filename = '{0}/{1}.{2}.pickle'.format(self.directory,self.funname,h)
        return filename

    def __setitem__(self, key, value):
        import pickle
        filename = self.get_filename(key)
        try:
            with open(filename,'w') as f:
                pickle.dump(value,f) 
        except TypeError as e:
            raise e
            
    def get(self, item):
        import pickle
        filename = self.get_filename(item)  
        try:
            with open(filename) as f:
                value = pickle.load(f)
                return value
        except :
            return None
    


import collections


def hashable(obj):
    if hasattr(obj,'flatten'): # for numpy arrays
        return tuple( obj.flatten().tolist() )
    if isinstance(obj, collections.Hashable):
        return obj
    if isinstance(obj, collections.Mapping):
        items = [(k,hashable(v)) for (k,v) in obj.items()]
        return frozenset(items)
    if isinstance(obj, collections.Iterable):
        return tuple([hashable(item) for item in obj])
    return TypeError(type(obj))
