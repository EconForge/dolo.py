from dolo import *

model = yaml_import("examples/models_/rbc.yaml")

print(model.calibration.flat)

time_iteration(model)