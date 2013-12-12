from __future__ import division

from dolo.symbolic.symbolic import Variable,Parameter,Shock,Equation
from dolo.symbolic.model import SModel
from collections import OrderedDict
import yaml
import sympy
import re

def iteritems(d):
    return zip(d.keys(), d.values())

def parse_yaml_text(txt,verbose=False, compiler=None):
    '''
Imports the content of a modfile into the current interpreter scope
'''
    txt = txt.replace('..','-')
    txt = txt.replace('--','-')
    txt = txt.replace('^','**')
    txt = txt.replace('equilibrium:','arbitrage:')
    txt = txt.replace('_|_','|')
    raw_dict = yaml.load(txt)

    if verbose == True:
        print('YAML file successfully parsed')

    declarations = raw_dict['declarations']
    # check
    variables_groups = OrderedDict()
    for vtype in declarations.keys():
        if vtype not in ('shocks','parameters'):
            variables_groups[vtype] = [Variable(vn) for vn in declarations[vtype]]
    variables_ordering = sum(variables_groups.values(),[])
#    else:
#        vnames = declarations['variables']
#        variables_ordering = [Variable(vn) for vn in vnames]
#        variables_groups = None

    parameters_ordering = [Parameter(vn) for vn in declarations['parameters']]
    shocks_ordering = [Shock(vn) for vn in declarations['shocks']]

    context = [(s.name,s) for s in variables_ordering + parameters_ordering + shocks_ordering]
    context = dict(context)

    from dolo.symbolic.symbolic import timeshift as TS


    # add some common functions
    for f in [sympy.log, sympy.exp,
              sympy.sin, sympy.cos, sympy.tan,
              sympy.asin, sympy.acos, sympy.atan,
              sympy.sinh, sympy.cosh, sympy.tanh,
              sympy.pi, sympy.sign]:
        context[str(f)] = f
    context['sqrt'] = sympy.sqrt

    context['TS'] = TS
    if 'horrible_hack' in raw_dict:
        tt = raw_dict['horrible_hack']
        exec(tt, context)


    import re
    # we recognize two kinds of equations:
    # lhs = rhs
    # lhs | comp where comp is a complementarity condition

    equations = []
    equations_groups = OrderedDict()
    raw_equations = raw_dict['equations']
    if not isinstance(raw_equations,dict):
        raw_dict['model_type'] = 'dynare'
        raw_equations = {'dynare_block': raw_equations}
    if True: # tests whether there are groups of equations
        for groupname in raw_equations.keys():
            equations_groups[groupname] = []
            for raw_eq in raw_equations[groupname]: # Modfile is supposed to represent a global model. TODO: change it
                teqg = raw_eq.split('|')
                teq = teqg[0]
                if '=' in teq:
                    lhs,rhs = str.split(teq,'=')
                else:
                    lhs = teq
                    rhs = '0'
                try:
                    lhs = eval(lhs,context)
                    rhs = eval(rhs,context)
                except Exception as e:
                    print('Error parsing equation : ' + teq)
                    print( str(e) )
                    raise e

                eq = Equation(lhs,rhs)
                eq.tag(eq_type=groupname)
                if len(teqg)>1:
                    comp = teqg[1]
                    eq.tag(complementarity=comp)
                equations.append(eq)
                equations_groups[groupname].append( eq )
    else:
        for teq in raw_equations:
            if '=' in teq:
                lhs,rhs = str.split(teq,'=')
            else:
                lhs = teq
                rhs = '0'
            try:
                lhs = eval(lhs,context)
                rhs = eval(rhs,context)
            except Exception as e:
                print('Error parsing equations : ' + teq)
                print(str(e))
            eq = Equation(lhs,rhs)
            equations.append(eq)
        equations_groups = None

    parameters_values = {}
    init_values = {}
    covariances = None
    calibration_str = {}

    if 'calibration' in raw_dict:
        calibration = raw_dict['calibration'].copy()
        if 'covariances' in calibration:
            cov_string = calibration.pop("covariances")
        else:
            cov_string = None
        for special in ('parameters', 'steady_state'):
          if special in calibration:
            new = calibration.pop(special)
            calibration.update(new)

    # now construct the symbolic calibration dictionaries
    parameters_values = {}
    init_values = {}
    for k in calibration:
      value = calibration[k]
      try:
        val = eval(str(value), context)
      except Exception as e:
        print("Error evaluating calibrated value for {}. Replacing by 'nan'".format(k))
        val = sympy.nan
      if Parameter(k) in parameters_ordering:
        skey = Parameter(k)
      else:
        skey = Variable(k)
      parameters_values[skey] = val

    if cov_string is not None:
            context['sympy'] = sympy
            covariances = eval('sympy.Matrix({0})'.format( cov_string ), context)
    else:
            covariances = None # to avoid importing numpy

    symbols = variables_groups

    symbols['shocks'] = shocks_ordering
    symbols['parameters'] = parameters_ordering

    calibration_s = {}
    calibration_s.update(parameters_values)
    calibration_s.update(init_values)

    from dolo.symbolic.model import SModel

    model = SModel( equations_groups, symbols, calibration_s, covariances )
    model.__data__ = raw_dict



    return model

def yaml_import(filename, verbose=False, recipes=None, compiler='numpy', **kwargs):
    '''Imports model defined in specified file'''

    import yaml
    if isinstance(recipes,str):
        with open(recipes) as f:
            recipes_d = yaml.load(f)
    else:
        recipes_d = recipes

    import os
    basename = os.path.basename(filename)
    fname = re.compile('(.*)\.yaml').match(basename).group(1)
    f = open(filename)
    txt = f.read()
    model = parse_yaml_text(txt,verbose=verbose, compiler=compiler)
    model.fname = fname
    model.name = fname

    if compiler is not None and model.__data__.get('model_type') != 'dynare':
        from dolo.compiler.compiler_python import GModel
        model = GModel(model, recipes=recipes_d, compiler=compiler, **kwargs)

        if model.model_type == 'fga':
          from dolo.compiler.converter import GModel_fg_from_fga
          model = GModel_fg_from_fga(model)
          import warnings
          warnings.warn("Models of type 'fga' are deprecated. Use type 'fg' instead.")
        return model


    return model
