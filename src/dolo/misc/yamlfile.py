from dolo.symbolic.symbolic import Variable,Parameter,Shock,Equation
from dolo.symbolic.model import Model
import numpy as np
import yaml

import sympy
import re
import inspect


def parse_yaml_text(txt):
    '''
    Imports the content of a modfile into the current interpreter scope
    '''
    txt = txt.replace('^','**')
    raw_dict = yaml.load(txt)

    declarations = raw_dict['declarations']
    # check
    if 'controls' in declarations:
        vnames = declarations['states'] + declarations['controls'] + declarations['expectations']
    else:
        vnames = declarations['variables']

    variables_ordering = [Variable(vn,0) for vn in vnames]
    parameters_ordering = [Parameter(vn) for vn in declarations['parameters']]
    print parameters_ordering
    shocks_ordering = [Shock(vn,0) for vn in declarations['shocks']]

    context = {s.name: s for s in variables_ordering + parameters_ordering + shocks_ordering}
    equations = []
    raw_equations = raw_dict['equations']
    if isinstance(raw_equations,dict):   # tests whether there are groups of equations
        for groupname in raw_equations.keys():
            for raw_eq in raw_equations[groupname]:
                if groupname == 'arbitrage':
                    teq,comp = raw_eq.split('|')
                    comp = str.strip(comp)
                else:
                    teq = raw_eq
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
                    print str(e)
                eq = Equation(lhs,rhs)
                eq.tag(eq_type=groupname)
                if groupname == 'arbitrage':
                    eq.tag(complementarity=comp)
                equations.append(eq)

    calibration = raw_dict['calibration']
    parameters_values = { Parameter(k): eval(str(v),context)   for  k,v in  calibration['parameters'].iteritems()  }
    #steady_state = raw_dict['steady_state']
    init_values = { Variable(vn,0): eval(str(value),context)   for  vn,value in  calibration['steady_state'].iteritems()  }

    covariances = eval('np.array({0})'.format( calibration['covariances'] ))
    print init_values
    print parameters_values

    model_dict = {
        'variables_ordering': variables_ordering,
        'parameters_ordering': parameters_ordering,
        'shocks_ordering': shocks_ordering,
        'equations': equations,
        'parameters_values': parameters_values,
        'init_values': init_values,
        'covariances': covariances

    }

    return Model(**model_dict)
                

def yaml_import(filename,names_dict={},full_output=False):
    '''Imports model defined in specified file'''
    import os
    basename = os.path.basename(filename)
    fname = re.compile('(.*)\.yaml').match(basename).group(1)
    f = file(filename)
    txt = f.read()
    model = parse_yaml_text(txt)
    model['name'] = fname
    return model