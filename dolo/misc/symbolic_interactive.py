from dolo.symbolic.symbolic import Variable,Parameter,Shock,IndexedSymbol

import inspect
import re

#def set_variables(s,names_dict={}):
#    """
#    Creates symbolic variable with the name *s*.
#    -- a string, either a single variable name, or
#    a space separated list of variable names, or
#    a list of variable names.
#    NOTE: The new variable is both returned and automatically injected into
#    the parent's *global* namespace. It's recommended not to use "var" in
#    library code, it is better to use symbols() instead.
#    EXEMPLES:
#    """
#
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        res = []
#        for t in s:
#            # skip empty stringG
#            if not t:
#                continue
#            if t in names_dict:
#                latex_name = names_dict[t]
#            else:
#                latex_name = None
#            sym = Variable(t,0,latex_name)
#            frame.f_globals[t] = sym
#            res.append(sym)
#        res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        frame.f_globals['variables'] = res
#        return res
#    finally:
#        del frame
#
#def add_variables(s,latex_names=None):
#    """
#    The same as set_variables but doesn't replace the existing variables.
#    """
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        if latex_names <> None:
#            sl = re.split(' ', latex_names)
#            if len(sl)<> len(s):
#                raise Exception, "You should supply one latex name per variable"
#        res = []
#        for i in range(len(s)):
#            t=s[i]
#            # skip empty stringG
#            if not t:
#                continue
#            if latex_names == None:
#                sym = Variable(t,0)
#            else:
#                sym = Variable(t,0,latex_name=sl[i])
#            frame.f_globals[t] = sym
#            res.append(sym)
#        res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        frame.f_globals['variables'] += res
#        return res
#    finally:
#        del frame
#
#def set_shocks(s,latex_names=None,names_dict={}):
#    """
#    Creates symbolic variable with the name *s*.
#    -- a string, either a single variable name, or
#    a space separated list of variable names, or
#    a list of variable names.
#    NOTE: The new variable is both returned and automatically injected into
#    the parent's *global* namespace. It's recommended not to use "var" in
#    library code, it is better to use symbols() instead.
#    EXAMPLES:
#    """
#
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        if latex_names <> None:
#            sl = re.split(' ', latex_names)
#            if len(sl)<> len(s):
#                raise Exception, "You should supply one latex name per variable"
#        res = []
#        for i in range(len(s)):
#            t = s[i]
#            # skip empty stringG
#            if not t:
#                continue
#            if latex_names != None:
#                sym = Shock(t,0,latex_name=sl[i])
#            elif t in names_dict:
#                    sym = Shock(t,0,latex_name=names_dict[t])
#            else:
#                sym = Shock(t,0)
#            frame.f_globals[t] = sym
#            res.append(sym)
#            res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        frame.f_globals['shocks'] = res
#        return res
#    finally:
#        del frame
#
#def add_shocks(s,latex_names=None):
#    """
#    The same as set_shocks but doesn't replace the existing variables.
#    """
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        if latex_names <> None:
#            sl = re.split(' ', latex_names)
#            if len(sl)<> len(s):
#                raise Exception, "You should supply one latex name per variable"
#        res = []
#        for i in range(len(s)):
#            t=s[i]
#            # skip empty stringG
#            if not t:
#                continue
#            if latex_names == None:
#                sym = Shock(t,0)
#            else:
#                sym = Shock(t,0,latex_name=sl[i])
#            frame.f_globals[t] = sym
#            res.append(sym)
#        res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        frame.f_globals['shocks'] += res
#        return res
#    finally:
#        del frame
#
#def set_parameters(s,names_dict={}):
#    """Create S symbolic variable with the name *s*.
#    -- a string, either a single variable name, or
#    a space separated list of variable names, or
#    a list of variable names.
#    NOTE: The new variable is both returned and automatically injected into
#    the parent's *global* namespace. It's recommended not to use "var" in
#    library code, it is better to use symbols() instead.
#    EXAMPLES:
#    """
#
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        res = []
#        for t in s:
#            # skip empty stringG
#            if not t:
#                continue
#            if t in names_dict:
#                latex_name = names_dict[t]
#            else:
#                latex_name = None
#            sym = Parameter(t,latex_name)
#            frame.f_globals[t] = sym
#            res.append(sym)
#        res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        frame.f_globals['parameters'] = res
#        return res
#    finally:
#        del frame
#
#def add_parameters(s,latex_names=None):
#    """
#    The same as set_variables but doesn't replace the existing variables.
#    """
#
#    frame = inspect.currentframe().f_back
#    try:
#        if not isinstance(s, list):
#            s = re.split('\s|,', s)
#        if latex_names <> None:
#            sl = re.split('\s|,', latex_names)
#            if len(sl)<> len(s):
#                raise Exception, "You should supply one latex name per variable"
#        res = []
#        for i in range(len(s)):
#            t=s[i]
#            # skip empty stringG
#            if not t:
#                continue
#            if latex_names == None:
#                sym = Parameter(t)
#            else:
#                sym = Parameter(t,latex_name=sl[i])
#            frame.f_globals[t] = sym
#            res.append(sym)
#            res = list(res)
#        if len(res) == 0: # var('')
#            res = []
#            # otherwise var('a b ...')
#        if frame.f_globals.get('parameters'):
#            frame.f_globals['parameters'].extend(res)
#        else:
#            frame.f_globals['parameters'] = res
#        return res
#    finally:
#        del frame

####  new style   #######

def def_variables(s):
    """
    blabla
    """

    frame = inspect.currentframe().f_back
    try:
        if isinstance(s,str):
            s = re.split('\s|,', s)
        res = []
        for t in s:
            # skip empty stringG
            if not t:
                continue
            if t.count("@") > 0:
                sym = IndexedSymbol(t,Variable)
                t = t.strip('@')
            else:
                sym = Variable(t)
            frame.f_globals[t] = sym
            res.append(sym)
        if frame.f_globals.get('variables_order'):
            # we should avoid to declare symbols twice !
            frame.f_globals['variables_order'].extend(res)
        else:
            frame.f_globals['variables_order'] = res
        return res
    finally:
        del frame

def def_shocks(s):
    """
    blabla
    """

    frame = inspect.currentframe().f_back
    try:
        if isinstance(s,str):
            s = re.split('\s|,', s)
        res = []
        for t in s:
            # skip empty stringG
            if not t:
                continue
            if t.count("@") > 0:
                sym = IndexedSymbol(t,Shock)
                t = t.strip('@')
            else:
                sym = Shock(t)
            frame.f_globals[t] = sym
            res.append(sym)
        if frame.f_globals.get('shocks_order'):
            # we should avoid to declare symbols twice !
            frame.f_globals['shocks_order'].extend(res)
        else:
            frame.f_globals['shocks_order'] = res
        return res
    finally:
        del frame


def def_parameters(s):
    """
    blabla
    """

    frame = inspect.currentframe().f_back
    try:
        if isinstance(s,str):
            s = re.split('\s|,', s)
        res = []
        for t in s:
            # skip empty stringG
            if not t:
                continue
            if t.count("@") > 0:
                sym = IndexedSymbol(t,Parameter)
                t = t.strip('@')
            else:
                sym = Parameter(t)
            frame.f_globals[t] = sym
            res.append(sym)
        if frame.f_globals.get('parameters_order'):
            # we should avoid to declare symbols twice !
            frame.f_globals['parameters_order'].extend(res)
        else:
            frame.f_globals['parameters_order'] = res
        return res
    finally:
        del frame


def clear_all():
    """
    Clears all parameters, variables, and shocks defined previously
    """

    frame = inspect.currentframe().f_back
    try:
        if frame.f_globals.get('variables_order'):
            # we should avoid to declare symbols twice !
            del frame.f_globals['variables_order']
        if frame.f_globals.get('parameters_order'):
            # we should avoid to declare symbols twice !
            del frame.f_globals['parameters_order']
    finally:
        del frame


def inject_symbols(symbs):
    frame = inspect.currentframe().f_back
    try:
        for s in symbs:
            sn = s.name
            frame.f_globals[sn] = s
    finally:
        del frame
