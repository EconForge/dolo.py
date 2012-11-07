from dolo import *

model = yaml_import('/home/pablo/Documents/Work/Florin/flexible.yaml')

from dolo.compiler.compiler_global import CModel
cm = CModel(model)

cm.x_bounds()

dr = global_solve(model, serial_grid=False, verbose=True)