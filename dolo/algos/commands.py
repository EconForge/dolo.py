def time_iteration(model, *args, **kwargs):

    if model.model_type in ('fg', 'fga'):
        from dolo.algos.fg.time_iteration import time_iteration
    elif model.model_type in ('mfg', 'mfga'):
        from dolo.algos.mfg.time_iteration import time_iteration
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return time_iteration(model, *args, **kwargs)

def approximate_controls(model, *args, **kwargs):

    if model.model_type in ('fg','fga'):
        order = kwargs.get('order')
        if order is None or order==1:
            from dolo.algos.fg.perturbations import approximate_controls
        else:
            from dolo.algos.fg.perturbations_higher_order import approximate_controls
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return approximate_controls(model, *args, **kwargs)

def simulate(model, *args, **kwargs):

    if model.model_type in ('fg','fga'):
        from dolo.algos.fg.simulations import simulate
    elif model.model_type in ('mfg','mfga'):
        from dolo.algos.mfg.simulations import simulate
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return simulate(model, *args, **kwargs)


def evaluate_policy(model, *args, **kwargs):

    if model.model_type in ('mfg','mfga'):
        from dolo.algos.mfg.value_iteration import evaluate_policy
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return evaluate_policy(model, *args, **kwargs)

def plot_decision_rule(model, *args, **kwargs):

    if model.model_type in ('fg','fga'):
        from dolo.algos.fg.simulations import plot_decision_rule
    elif model.model_type in ('mfg','mfga'):
        from dolo.algos.mfg.simulations import plot_decision_rule
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return plot_decision_rule(model, *args, **kwargs)

from dolo.misc.decorators import deprecated

@deprecated
def global_solve(*args, **kwargs):
    return time_iteration(*args, **kwargs)
