'''
This module contains the officially supported commands of Dolo :
- get_current_model()
- print_model()
- stoch_simul()
- irf()
'''

import matplotlib
import pylab
import dolo
import numpy

from pylab import *
    
from dolo.numeric.decision_rules import impulse_response_function as irf
from dolo.numeric.decision_rules import stoch_simul
from dolo.numeric.perturbations import solve_decision_rule


def dolo_help():
    txt = """
    Interactive system not available yet.

    Models can be written in a cell with %modfile(fname='yourname') prefix
    or imported from a modfile using dynare_import command.

    Dolo commands :
    (additional help can be obtained by taping cmd? + Tab)

    - dynare_import : import a model from a Dynare modfile.
    - print_model : print model with equations residuals.
    - solve_dr : compute decision rule.
    - irf : compute/plot impulse-response functions.
    - stoch_simul : compute/plot stochastic simulations.
    - pprint : prints a model or a decision rule
"""
    print(txt)


def get_current_model():
    try:
        model = globals()['dynare_model']
        return model
    except:
        raise( Exception( "You need to define a model first !" ) )


from dolo.config import display