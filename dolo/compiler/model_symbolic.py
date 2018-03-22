from .recipes import recipes


class SymbolicModel:
    def __init__(self,
                 model_name,
                 model_type,
                 symbols,
                 equations,
                 calibration,
                 domain=None,
                 exogenous=None,
                 options=None,
                 definitions=None):

        self.name = model_name
        self.model_type = model_type

        # reorder symbols
        canonical_order = [
            'variables', 'exogenous', 'states', 'controls', 'values', 'shocks',
            'parameters'
        ]
        osyms = dict()
        for vg in canonical_order:
            if vg in symbols:
                osyms[vg] = symbols[vg]
        for vg in symbols:
            if vg not in canonical_order:
                osyms[vg] = symbols[vg]

        self.symbols = osyms
        self.equations = equations
        self.calibration_dict = calibration

        self.domain = domain
        self.exogenous = exogenous
        self.options = options
        self.definitions = definitions
