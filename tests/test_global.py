from dolo import *

from dolo.misc.yamlfile import yaml_import

model = yaml_import('../examples/global_models/optimal_growth.yaml')

from dolo.compiler.compiler_mirfac import MirFacCompiler

comp = MirFacCompiler(model)

#txt = comp.process_output_python()
#print txt

txt = comp.process_output_matlab()
print txt