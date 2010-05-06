# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="pablo"
__date__ ="$3 avr. 2010 16:14:59$"

from dolo import *
from dolo.compiler.compiler_dynare import DynareCompiler

resp = dynare_import('../examples/dynare_modfiles/example1.mod')

model = resp['model']
model.name = 'example1'
model.check()
#

comp = DynareCompiler(model)
print comp.lead_lag_incidence_matrix()

f = comp.compute_dynamic_pfile(2)
#comp.export_to_modfile('test.mod')

import numpy as np

y = np.zeros(14)
x = np.zeros((1,2))
params = np.zeros(7)
res =  f(y,x,params)
[g0,g1,g2] = res

print g0
print g1
#model = dynare_model['model']