import numpy
import ruamel.yaml as ry

from dolo.misc.display import read_file_or_url
import yaml
from collections import OrderedDict

def yaml_import(fname, check=True, check_only=False):

    txt = read_file_or_url(fname)

    if check:
        from dolo.linter import lint
        output = lint(txt)
        if len(output)>0:
            print(output)

    if check_only:
        return output

    txt = txt.replace('^', '**')

    data = ry.load(txt, ry.RoundTripLoader)
    data['filename'] = fname

    from dolo.compiler.model import Model

    return Model(data)


# This should be done in ModelSymbolic
#
# # all symbols are initialized to nan
# # except shocks and markov_states which are initialized to 0
# initial_values = {
#     'shocks': 0,
#     # 'markov_states': 0,
#     'exogenous': 0,
#     'expectations': 0,
#     'values': 0,
#     'controls': float('nan'),
#     'states': float('nan')
# }
#
# # variables defined by a model equation default to using these definitions
# initialized_from_model = {
#     'values': 'value',
#     'expectations': 'expectation',
#     'direct_responses': 'direct_response'
# }
#
# for k, v in definitions.items():
#     if k not in calibration:
#         calibration[k] = v
#
# for symbol_group in symbols:
#     if symbol_group not in initialized_from_model.keys():
#         if symbol_group in initial_values:
#             default = initial_values[symbol_group]
#         else:
#             default =  float('nan')
#         for s in symbols[symbol_group]:
#             if s not in calibration:
#                 calibration[s] = default





if __name__ == "__main__":

    # fname = "../../examples/models/compat/rbc.yaml"
    fname = "examples/models/compat/integration_A.yaml"

    import os
    print(os.getcwd())

    model = yaml_import(fname)

    print("calib")
    # print(model.calibration['parameters'])

    print(model)

    print(model.get_calibration(['beta']))
    model.set_calibration(beta=0.95)

    print( model.get_calibration(['beta']))


    print(model)

    s = model.calibration['states'][None,:]
    x = model.calibration['controls'][None,:]
    e = model.calibration['shocks'][None,:]

    p = model.calibration['parameters'][None,:]

    S = model.functions['transition'](s,x,e,p)
    lb = model.functions['controls_lb'](s,p)
    ub = model.functions['controls_ub'](s,p)


    print(S)

    print(lb)
    print(ub)
