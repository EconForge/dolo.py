from .recipes import recipes



class SymbolicModel:

    def __init__(self, model_name, model_type, symbols, symbolic_equations, symbolic_calibration,
                    options=None, definitions=None):

        self.name = model_name
        self.model_type = model_type

        # reorder symbols
        from collections import OrderedDict
        canonical_order = ['variables', 'markov_states', 'states', 'controls', 'values', 'shocks', 'parameters']
        osyms = OrderedDict()
        for vg in canonical_order:
            if vg in symbols:
                 osyms[vg] = symbols[vg]
        for vg in symbols:
            if vg not in canonical_order:
                 osyms[vg] = symbols[vg]

        self.symbols = osyms
        self.equations = symbolic_equations
        self.calibration_dict = symbolic_calibration

        # self.distribution = distribution
        # self.discrete_transition = discrete_transition

        self.options = options
        self.definitions = definitions

        self.check()

    def check(self):

        # reuse code from linter ?
        pass
