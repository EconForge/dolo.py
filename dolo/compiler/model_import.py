import numpy

from dolo.misc.display import read_file_or_url
import yaml
from collections import OrderedDict

def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):

    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# usage example:

def yaml_import(fname, return_symbolic=False, check=True, check_only=False):

    txt = read_file_or_url(fname)

    if check:
        from dolo.linter import lint
        output = lint(txt)
        if len(output)>0:
            print(output)

    if check_only:
        return output

    txt = txt.replace('^', '**')

    return fast_import(txt, return_symbolic=return_symbolic, filename=fname)


def autodetect_type(data):
    if 'variables' in data['symbols']: return 'dynare'
    elif 'markov_states' in data['symbols']: return 'dtmscc'
    else: return 'dtcscc'


def fast_import(txt, return_symbolic=False, filename='<string>', parse_only=False):

    import yaml
    from dolo.compiler.language import minilang

    for C in minilang:
        k = C.__name__
        yaml.add_constructor('!{}'.format(k), C.constructor)

    txt = txt.replace('^', '**')

    data = yaml.load(txt)
    if parse_only:
        return data

    name = data['name']

    model_type = data.get('model_type')
    auto_type = autodetect_type(data)

    if model_type is None:
        model_type = auto_type
        print("Missing `model_type` field. Set to `{}`".format(auto_type))
    else:
        assert(model_type == auto_type)
    symbols = data['symbols']
    definitions = data.get('definitions', {})
    equations = data['equations']
    calibration = data.get('calibration', {})
    options = data.get('options', {})

    infos = dict()
    infos['filename'] = filename
    infos['name'] = name
    infos['type'] = model_type

    # all symbols are initialized to nan
    # except shocks and markov_states which are initialized to 0
    initial_values = {
        'shocks': 0,
        'markov_states': 0,
        'expectations': 0,
        'values': 0,
        'controls': float('nan'),
        'states': float('nan')
    }

    # variables defined by a model equation default to using these definitions
    initialized_from_model = {
        'values': 'value',
        'expectations': 'expectation',
        'direct_responses': 'direct_response'
    }

    for k, v in definitions.items():
        if k not in calibration:
            calibration[k] = v

    for symbol_group in symbols:
        if symbol_group not in initialized_from_model.keys():
            if symbol_group in initial_values:
                default = initial_values[symbol_group]
            else:
                default =  float('nan')
            for s in symbols[symbol_group]:
                if s not in calibration:
                    calibration[s] = default

    from dolo.compiler.model_symbolic import SymbolicModel
    smodel = SymbolicModel(name, model_type, symbols, equations,
                           calibration, options=options, definitions=definitions)

    if return_symbolic:
        return smodel

    if model_type in ('dtcscc', 'dtmscc'):
        from dolo.compiler.model_numeric import NumericModel
        model = NumericModel(smodel, infos=infos)
    else:
        from dolo.compiler.model_dynare import DynareModel
        model = DynareModel(smodel, infos=infos)
    return model


if __name__ == "__main__":

    # fname = "../../examples/models/rbc.yaml"
    fname = "examples/models/integration_A.yaml"

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


    # print(model.calibration['parameters'])
