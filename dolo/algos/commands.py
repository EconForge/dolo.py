def time_iteration(model, *args, **kwargs):

    if model.model_type == 'dtcscc':
        from dolo.algos.dtcscc.time_iteration import time_iteration
    elif model.model_type == 'dtmscc':
        from dolo.algos.dtmscc.time_iteration import time_iteration
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return time_iteration(model, *args, **kwargs)

def approximate_controls(model, *args, **kwargs):

    if model.model_type == 'dtcscc':
        order = kwargs.get('order')
        if order is None or order==1:
            from dolo.algos.dtcscc.perturbations import approximate_controls
        else:
            from dolo.algos.dtcscc.perturbations_higher_order import approximate_controls
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return approximate_controls(model, *args, **kwargs)

def simulate(model, *args, **kwargs):

    if model.model_type == 'dtcscc':
        from dolo.algos.dtcscc.simulations import simulate
    elif model.model_type == 'dtmscc':
        from dolo.algos.dtmscc.simulations import simulate
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return simulate(model, *args, **kwargs)


def evaluate_policy(model, *args, **kwargs):

    if model.model_type == 'dtmscc':
        from dolo.algos.dtmscc.value_iteration import evaluate_policy
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return evaluate_policy(model, *args, **kwargs)

def plot_decision_rule(model, *args, **kwargs):

    if model.model_type == 'dtcscc':
        from dolo.algos.dtcscc.simulations import plot_decision_rule
    elif model.model_type == 'dtmscc':
        from dolo.algos.dtmscc.simulations import plot_decision_rule
    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return plot_decision_rule(model, *args, **kwargs)


def perfect_foresight(model, *args, **kwargs):

    if model.model_type == 'dtcscc':
        from dolo.algos.dtcscc.perfect_foresight import deterministic_solve

    else:
        raise Exception("Model type {} not supported.".format(model.model_type))

    return deterministic_solve(model, *args, **kwargs)
