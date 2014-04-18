from recipes import recipes



class SymbolicModel:

    def __init__(self, model_name, model_type, symbols, symbolic_equations, symbolic_calibration,
                 symbolic_covariances=None, symbolic_markov_chain=None, options=None):

        self.name = model_name
        self.model_type = model_type
        self.symbols = symbols
        self.equations = symbolic_equations
        self.calibration_dict = symbolic_calibration
        self.covariances = symbolic_covariances
        self.markov_chain = symbolic_markov_chain
        self.options = options

        self.check()

    def check(self):

        if self.model_type == 'fg':

            n_eq_transition = len(self.equations['transition'])
            n_eq_arbitrage = len(self.equations['arbitrage'])

            assert( len(self.symbols['states']) == n_eq_transition)
            assert( len(self.symbols['controls']) == n_eq_arbitrage)

            if 'auxiliary' in self.equations:
                n_eq_auxiliary = len(self.equations['auxiliary'])
                assert( len(self.symbols['auxiliaries']) == n_eq_auxiliary)

        else:
            pass
            # raise Exception( "No rule to check model type {}".format(self.model_type))


