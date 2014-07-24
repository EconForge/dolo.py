def time_iteration(model, *args, **kwargs):

    if model.model_type == 'fga':
        from dolo.algos.fg.time_iteration import time_iteration
    elif model.model_type == 'mfg':
        from dolo.algos.mfg.time_iteration import time_iteration
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return time_iteration(model, *args, **kwargs)


def simulate(model, *args, **kwargs):

    if model.model_type == 'fga':
        from dolo.algos.fg.simulations import simulate
    elif model.model_type == 'mfg':
        from dolo.algos.mfg.simulations import simulate
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return simulate(model, *args, **kwargs)

def plot_decision_rule(model, *args, **kwargs):

    if model.model_type == 'fga':
        from dolo.algos.fg.simulations import plot_decision_rule
    elif model.model_type == 'mfg':
        from dolo.algos.mfg.simulations import plot_decision_rule
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return plot_decision_rule(model, *args, **kwargs)
