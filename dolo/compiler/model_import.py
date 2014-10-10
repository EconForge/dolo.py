from __future__ import division

import numpy

from dolo.misc.display import read_file_or_url

def yaml_import(fname, txt=None, return_symbolic=False):

    if txt is None:

        txt = read_file_or_url(fname)

    txt = txt.replace('^', '**')

    import yaml

    try:
        data = yaml.safe_load(txt)
    except Exception as e:
        raise e

    if 'symbols' not in data:
        if 'declarations' in data:
            data['symbols'] = data['declarations']
            # TODO: raise an error/warning here
        else:
            raise Exception("Missing section: 'symbols'.")

    if 'model_type' not in data:
        if 'markov_states' in data['symbols']:
            if 'auxiliaries' in data['symbols']:
                model_type = 'mfga'
            else:
                model_type = 'mfg'
        elif 'states' in data['symbols']:
            if 'auxiliaries' in data['symbols']:
                model_type = 'fga'
            else:
                model_type = 'fg'
        elif 'variables' in data['symbols']:
            model_type = 'dynare'
        else:
            raise Exception("'model_type' was not defined and couldn't be guessed.")
        print("Model type detected as {}".format(model_type))
    else:
        model_type = data['model_type']


    if 'name' not in data:
        data['name'] = 'anonymous'
        print("Missing model name. Set as '{}'".format(data['name']))



    if 'auxiliary' in data['symbols']:
        aux = data['symbols'].pop('auxiliary')
        data['symbols']['auxiliaries'] = aux

    # check equations
    if 'equations' not in data:
        raise Exception("Missing section: 'equations'.")

    if 'calibration' not in data:
        raise Exception("Missing section: 'calibration'.")

    options = data.get('options')

    if 'steady_state' in data['calibration']:
        oldstyle = data['calibration']
        covs = oldstyle['covariances']
        steady = oldstyle['steady_state']
        params = oldstyle['parameters']
        pp = dict()
        pp.update(steady)
        pp.update(params)
        data['calibration'] = pp

        data['covariances'] = eval(
            "numpy.array({}, dtype='object')".format(covs)
        )

    # model specific

    if model_type in ('fga', 'fgh', 'vfi', 'dynare'):
        if 'covariances' not in data:
            raise Exception(
                "Missing section (model {}): 'covariances'.".format(model_type)
            )
        symbolic_covariances = data['covariances']

    if model_type in ('mfg', 'mfga', 'mvfi'):
        if 'markov_chain' not in data:
            raise Exception(
                "Missing section (model {}): 'markov_chain'.".format(model_type)
            )
        symbolic_markov_chain = data['markov_chain']

    model_name = data['name']
    symbols = data['symbols']
    symbolic_equations = data['equations']
    symbolic_calibration = data['calibration']

    # all symbols are initialized to nan
    # except shocks and markov_states which are initialized to 0
    initial_values = {
        'shocks': 0,
        'markov_states': 0,
        'controls': float('nan'),
        'states': float('nan')
    }

    for symbol_group in symbols:
        if symbol_group in initial_values:
            default = initial_values[symbol_group]
        else:
            default = float('nan')
        for s in symbols[symbol_group]:
            if s not in symbolic_calibration:
                symbolic_calibration[s] = default


    # read covariance matrix

    symbolic_covariances = data.get('covariances')

    if symbolic_covariances is not None:
        try:
            tl = numpy.array(symbolic_covariances, dtype='object')
        except:
            msg = "Incorrect covariances matrix: {}.".format(
                symbolic_covariances
            )
            raise Exception(msg)
        try:
            assert(tl.ndim == 2)
            assert(tl.shape[0] == tl.shape[1])
        except:
            msg = """Covariances matrix should be square.\
            Found {} matrix""".format(tl.shape)
            raise Exception(msg)
        symbolic_covariances = tl

    symbolic_markov_chain = data.get('markov_chain')
    # TODO: read markov chain

    definitions = data.get('definitions', {})


    options = data.get('options')

    infos = dict()
    infos['filename'] = fname
    infos['name'] = model_name
    infos['type'] = model_type

    from dolo.compiler.model_symbolic import SymbolicModel
    smodel = SymbolicModel(model_name, model_type, symbols, symbolic_equations,
                           symbolic_calibration, symbolic_covariances,
                           symbolic_markov_chain,
                           options=options, definitions=definitions)

    if return_symbolic:
        return smodel

    if model_type in ('fg','fga','mfg', 'mfga'):
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
    lb = model.functions['arbitrage_lb'](s,p)
    ub = model.functions['arbitrage_ub'](s,p)


    print(S)

    print(lb)
    print(ub)


    # print(model.calibration['parameters'])
