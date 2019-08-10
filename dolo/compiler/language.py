import warnings
import copy
import numpy
import math
import ruamel.yaml as ry
from ruamel.yaml.comments import CommentedMap, CommentedSeq
import numpy as np
from numpy import ndarray
from dataclasses import dataclass

functions = {
    'log': math.log,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
}

constants = {
    'pi': math.pi
}

# this is a stupid implementation
class Language:

    functions = functions
    constants = constants
    objects = []
    object_names = []
    yaml_tags = []
    signatures = []

    def append(self, obj):

        self.objects.append(obj)
        self.object_names.append(obj.__name__)
        self.yaml_tags.append('!'+obj.__name__)

        # try to get signature
        try:
            sig = obj.signature
        except:
            sig = None
        self.signatures.append(sig)

    def isvalid(self, name):

        if name[0]=='!':
            name = name[1:]

        return (name in self.object_names)

    def get_from_tag(self,tag):
        assert(tag[0]=='!')
        i = self.object_names.index(tag[1:])
        obj = self.objects[i]
        return obj

    def get_signature(self, tag):
        i = self.object_names.index(tag[1:])
        obj = self.objects[i]
        try:
            sig = obj.signature
        except:
            sig = None
        return sig

LANG = Language()

def language_element(el):
    LANG.append(el)
    return el



class ModelError(Exception):
    pass


def eval_data(data: 'ruamel_structure', calibration={}):



    if isinstance(data, (float, int)):
        return data

    elif isinstance(data, str):
        # could be a string, could be an expression, could depend on other sections
        pass

    elif isinstance(data, CommentedSeq):

        tag = data.tag
        if tag.value is not None and not LANG.isvalid(tag.value):
            # unknown object type
            lc = data.lc
            msg = f"Line {lc.line}, column {lc.col}.  Tag '{tag.value}' is not recognized.'"
            raise ModelError(msg)

        # eval children
        children = [eval_data(ch, calibration) for ch in data]

        if tag.value is None:
            return children
        else:
            objclass = LANG.get_from_tag(tag.value)
            return objclass(*children)

    elif isinstance(data, CommentedMap):


        if data.tag is not None and data.tag.value=='!Function':
            return eval_function(data, calibration)

        tag = data.tag
        if tag.value is not None and not LANG.isvalid(tag.value):
            # unknown object type
            lc = data.lc
            msg = f"Line {lc.line}, column {lc.col}.  Tag '{tag.value}' is not recognized.'"
            raise ModelError(msg)

        if tag.value is not None:
            # check argument names (ignore types for now)
            objclass = LANG.get_from_tag(tag.value)
            signature = LANG.get_signature(tag.value)
            sigkeys =  [*signature.keys()]
            for a in data.keys():
                ## TODO account for repeated greek arguments
                if (a not in sigkeys) and (greek_translation.get(a,None) not in sigkeys):
                    lc = data.lc
                    sigstring = str.join(', ', [f"{k}={str(v)}" for k,v in signature.items()])
                    msg = f"Line {lc.line}, column {lc.col}. Unexpected argument '{a}'. Expected: '{objclass.__name__}({sigstring})'"
                    raise ModelError(msg)
                else:
                    try:
                        sigkeys.remove(a)
                    except:
                        sigkeys.remove(greek_translation[a])
            # remove optional arguments
            for sig in sigkeys:
                sigval = signature[sig]
                if sigval is not None and ('Optional' in sigval):
                    sigkeys.remove(sig)

            if len(sigkeys)>0:
                sigstring = str.join(', ', [f"{k}={str(v)}" for k,v in signature.items()])
                lc = data.lc
                msg = f"Line {lc.line}, column {lc.col}. Missing argument(s) '{str.join(', ',sigkeys)}'. Expected: '{objclass.__name__}({sigstring})'"
                raise ModelError(msg)


        # eval children
        children = []
        for key, ch in data.items():
            evd = eval_data(ch, calibration=calibration)
            if tag.value is not None:
                exptype = signature.get(key, None)
                if exptype in ['Matrix', 'Optional[Matrix]']:
                    matfun = LANG.get_from_tag('!Matrix')
                    try:
                        evd = matfun(*evd)
                    except:
                        lc = data.lc
                        msg = f"Line {lc.line}, column {lc.col}. Argument '{key}' could not be converted to Matrix"
                        raise ModelError(msg)
                elif exptype in ['Vector', 'Optional[Vector]']:
                    vectfun = LANG.get_from_tag('!Vector')
                    try:
                        evd = vectfun(*evd)
                    except:
                        lc = data.lc
                        msg = f"Line {lc.line}, column {lc.col}. Argument '{key}' could not be converted to Vector"
                        raise ModelError(msg)
            children.append(evd)

        kwargs = {k: v for (k,v) in zip(data.keys(), children)}
        if tag.value is None:
            return kwargs
        else:
            objclass = LANG.get_from_tag(tag.value)
            return objclass(**kwargs)

    if isinstance(data, str):
        try:
            val = eval(data, calibration)
        except:
            warnings.warn("Impossible to evaluate expression")
            val = data

        return val

def eval_function(data, calibration):
    args = tuple(data['arguments'])
    content = copy.deepcopy(data['value'])
    def fun(x):
        calib = calibration.copy()
        for i, a in enumerate(args):
            calib[a] = x[i]
        res = eval_data(content, calib)
        return res
    return fun





# GREEK TOLERANCE

greek_translation = {
    'Sigma': 'Σ',
   'sigma': 'σ',
   'rho': 'ρ',
   'mu': 'μ',
   'alpha': 'α',
   'beta': 'β'
}

def greekify_dict(arg):
   dd = dict()
   for k in arg:
       if k in greek_translation:
           key = greek_translation[k]
       else:
           key = k
       if key in dd:
           raise Exception(f"key {key} defined twice")
       dd[key] = arg[k]
   return dd


def greek_tolerance(fun):

   def f(*pargs, **args):
       nargs = greekify_dict(args)
       return fun(*pargs, **nargs)

   return f
