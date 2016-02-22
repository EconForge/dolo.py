from collections import OrderedDict

import numexpr
import numpy as np

from dolo.compiler.recipes import recipes
from dolo.compiler.function_compiler_ast import compile_function_ast


def _try_get(d, k, part="aggregation"):
    x = d.get(k, None)
    if x is None:
        msg = "{0} is requried for {1} specification, but missing"
        raise ValueError(msg.format(k, part))

    return x


def _strip_equal(eq):
    lhs, rhs = str.split(eq,'=')
    eq = '{} - ( {} )'.format(rhs.strip(), lhs.strip())
    return str.strip(eq)


class Population(object):
    """
    Describes a population of heterogenous entities in a model

    Attributes
    ----------
    name: str
        A string describing the name of the population

    problem: str
        A string specifying the name of the model that describes the
        individual problem

    indices: dict(str,int)
        Maps index names to integers specifying the number of elements
        in the discretization of that index

    n_{x}: int
        Integers of the form `n_x` where x is in the keys of indices.
        Just a convenience repetition of the data in indices

    """

    def __init__(self, name, data):
        self.name = name
        self.problem_name = _try_get(data, "problem", "population")
        self.indices = {}

        ix = _try_get(data, "index", "population")

        for k in ix.keys():
            self.__setattr__("n_{0}".format(k), ix[k])
            self.indices[k] = ix[k]

    def _attach_problem(self, problems):
        for p in problems:
            if p.name == self.problem_name:
                self.problem = p
                break
        else:
            msg = "Problem named {0} not found in list of individual problems"
            raise ValueError(msg.format(self.problem_name))


class ModelAggregation(object):
    """
    Describes the aggregation conditions in a heterogenous agent model

    Fields
    ------
    free_parameters: list(str)
        A list of strings containing the symbol names for parameters
        that will adjust to satisfy equilibrium conditions
    distributions: dict(str,array)
        A dict that maps parameters
    eqm_conditions: list(function)
        A list of functions that specify the equilibirum conditions of
        the model. Should be of the form `something = 0`.

    """

    def __init__(self, data, problems=None):
        if problems is None:
            msg = """Cannot compile functions for equilibrium conditions\
            without access to individual problems
            """

        if len(problems) != 1:
            raise ValueError("Right now we only support one individual problem")

        # For now we will enforce that `model_type: aggregation` has been set
        self.model_type = _try_get(data, "model_type")
        if self.model_type != "aggregation":
            raise ValueError("model_type must be aggregation")

        # TODO: might want to get rid of _data
        self._data = data

        # extract free parameters and populations.
        # TODO: parse the bounds
        self.free_parameters = _try_get(data, "free_parameters")
        self.populations = [Population(k, v) for (k, v) in
                            _try_get(data, "population").items()]

        for pop in self.populations:
            pop._attach_problem(problems)

        # bring all indices to the top level
        self.indices = {}
        for pop in self.populations:
            self.indices.update(pop.indices)

        # now extract distributions. This will map from keys that are parameter
        # values in the individual problems to values that specify the levels
        # of that parameter
        self.distributions = {k: self._parse_distribution(v) for (k, v) in
                              _try_get(data, "distribution").items()}

        self._compile_eqm_conditions()

    def _parse_distribution(self, v):
        ind, expr = list(map(str.strip, v.split("->")))
        n_ind = "n_{0}".format(ind)

        # TODO: this is super unsafe, but it is a prototype after all ;)
        context = {ind: np.arange(1, self.indices[ind] + 1),
                   n_ind: self.indices[ind]}
        return eval(expr, context, context)

    def _compile_eqm_conditions(self):
        # TODO: Lots of work to be done here.
        eqs = _try_get(self._data, "equilibrium")
        eqs = [_strip_equal(eq) for eq in eqs]

        recipe = recipes["aggregation"]
        funname = "equilibrium"
        target_spec = None
        arg_names = recipe["specs"][funname]["eqs"]

        # extract data from the underlying individual problem
        symbols = self.populations[0].problem.symbols
        defs = self.populations[0].problem.symbolic.definitions
        auxdefs = self.populations[0].problem._auxdefs

        # deal with
        ddefs = OrderedDict()
        for ag in arg_names:
            if ag[0] == 'auxiliaries':
                t = ag[1]
                ddefs.update(auxdefs[t])
        ddefs.update(defs)

        pyfun, fun = compile_function_ast(eqs, symbols, arg_names,
                                   output_names=target_spec,
                                   funname=funname,
                                   definitions=ddefs,
                                   print_code=True,
                                   vectorize=False,
                                   original=True)

        self.function = fun
        self._py_fun = pyfun

if __name__ == '__main__':
    from dolo import *
    import yaml
    m = yaml_import("../../examples/models/bewley_dtcscc.yaml")
    txt = open("../../examples/models/bewley_aggregate.yaml").read()
    data = yaml.safe_load(txt)
    problems = [m]
    ma = ModelAggregation(data, problems)
