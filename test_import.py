import dolo.config
dolo.config.debug = True
import ruamel.yaml as ry
from ruamel.yaml.comments import CommentedSeq, CommentedMap

from dolo.compiler.model import Model
from dolo import yaml_import, time_iteration

fname = "examples/models/rbc0.yaml"
model = yaml_import(fname)
print(model)

time_iteration(model)

# with open("examples/models/rbc0.yaml") as f:
#     txt = f.read()
#
# print( txt )
#
# txt = txt.replace("^","**")
#
# data = ry.load(txt, ry.RoundTripLoader)
# data['filename'] = "examples/models/rbc00.yaml"
#
#
# model = Model(data)
# print(model.definitions)
# print(model.name)
# print(model.model_type)
# print("Calibration")
# print(model.calibration['states'])
# print(model.domain)
#
# m,s,x,p = model.calibration['exogenous','states','controls','parameters']
#
#
# S = model.functions['transition'](m,s,x,m,p)
#
# print( S )
#
# y = model.functions['auxiliary'](m,s,x,p)
# print(y)
#
# print(model)
#
# from dolo import time_iteration
# time_iteration(model, verbose=True)
# print(model._repr_html_())
