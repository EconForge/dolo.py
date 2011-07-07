from dolo import *
from dolo.misc.yamlfile import *
model = yaml_import('/home/pablo/Temp/BGM.yaml')

dr = solve_decision_rule(model)
print dr