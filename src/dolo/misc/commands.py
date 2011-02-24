'''
This module contains the officially supported commands of Dolo :
- get_current_model()
- print_model()
- stoch_simul()
- irf()
'''

import pylab
import dolo
import numpy

from pylab import *

try:
    import sage
    __sage_instance__ = sage
    __sage_is_running__ = True
    #del sage
    #del record
except Exception as e:
    __sage_is_running__ = False

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
    if __sage_is_running__:
        from sage.misc.html import HTML
        html = HTML()
        html(txt)



def get_current_model():
    try:
        model = globals()['dynare_model']
        return model
    except:
        raise( Exception( "You need to define a model first !" ) )

def print_model(model=None, print_residuals=True):

    if not model:
        model = get_current_model()
        
    if __sage_is_running__:
        from sage.misc.html import HTML
        html = HTML()
        if print_residuals:
            from dolo.symbolic.model import compute_residuals
            res = compute_residuals(model)
            #res = [0]*len(model.equations)
            html.table([['','Equations','Residuals']] + [(i+1,model.equations[i],"%.4f" %float(res[i])) for i in range(len(model.equations))],header=True)
        else:
            html.table([['','Equations']] + [(i+1,model.equations[i]) for i in range(len(model.equations))], header=True)
    else:
        print( str(model) )

def print_table( tab,precision=3, col_names=None, row_names=None ):
    if __sage_is_running__:
        from sage.misc.html import HTML
        html = HTML()
    else:
        raise Exception('Sage is not running.')
    import numpy as np
    resp = [[ "%.3f" %tab[i,j] for j in range(tab.shape[1]) ] for i in range(tab.shape[0]) ]
    if row_names:
        resp = [  [row_names[i]] + resp[i] for i in range(tab.shape[0]) ]
    header = False
    if col_names:
        resp = [[''] +col_names] + resp
        header = True
    HTML.table(resp,header)

def pprint(obj,col_names=None,row_names=None):
    if __sage_is_running__:
        from sage.misc.html import HTML
        html = HTML()
    if isinstance(obj,dolo.symbolic.model.Model):
        print_model(obj)
    elif isinstance(obj,dolo.numeric.decision_rules.DecisionRule):
        dr = obj
        model = dr.model
        states = model.state_variables
        states_i = [model.variables.index(v) for v in states]
        mat = numpy.concatenate( [dr['g_a'][:,states_i], dr['g_e']],axis=1 )
        tab = [['%.3f' %n for n in  mat[i,:]] for i in range(len(model.variables)) ]
        tab = [ [v]+tab[i] for i,v in enumerate(model.variables) ]
        tab = [['']+[s(-1) for s in states]+model.shocks] + tab
        html.table(tab,header=True)
    elif isinstance(obj,numpy.ndarray):
        tab = obj
        resp = [[ "%.3f" %tab[i,j] for j in range(tab.shape[1]) ] for i in range(tab.shape[0]) ]
        if row_names:
            resp = [  [row_names[i]] + resp[i] for i in range(tab.shape[0]) ]
        header = False
        if col_names:
            resp = [[''] +col_names] + resp
            header = True
        html.table(resp,header)


    else:
        print'Object type not known.'

def simulated_moments(decision_rule):
    dr = decision_rule
    


class ModFileCell():

    def __init(self):
        self.fname = 'anonymous'

    def __call__(self,**args):
        self.fname = args.get('fname') if args.get('fname')  else 'anonymous'
        return self

    def eval(self,s,d,locals={}):
        s = s.encode() # this is to avoid incompatibilities if cell contains unicode
        #locals['dynare_modfile'] = s
        DATA = locals['DATA']

        fname = self.fname

        f = file(DATA + fname + '.mod' ,'w')
        f.write(s)
        f.close()

        print "Modfile has been written as : '{0}.mod'\n".format(fname)

        #try:
        from dolo.misc.modfile import parse_dynare_text
        from dolo.symbolic.model import Model

        t = parse_dynare_text(s,full_output=True)
        t['name']=fname
        dynare_model = Model(**t)
        dynare_model.check_consistency(verbose=True)

        globals()['dynare_model'] = dynare_model

        import inspect
        nb_frame = inspect.getouterframes(inspect.currentframe().f_back)[1][0]
        nb_frame.f_globals['dynare_model'] = dynare_model

        print '\nDynare model successfully parsed and stored in "dynare_model"\n'
#      except NameError:
#            print 'Dolo is not installed'
#            None
            # do nothing : Dolo is not installed
        #except Exception as exc:
        #    print 'Dolo was unable to parse dynare block.'
        #    print exc

        return ''

modfile = ModFileCell()

if not __sage_is_running__:
    # these functions make no sense
    del modfile
    del ModFileCell
