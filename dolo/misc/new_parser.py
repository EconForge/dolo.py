recipes = {

    'fg': {

        'symbols': ['states', 'controls', 'shocks', 'parameters'],

        'specs': {

            'transition': [
                ('states', -1, 's'),
                ('controls', -1, 'x'),
                ('shocks', 0, 'e'),
                ('parameters', 0, 'p')
            ],


            'arbitrage': [
                ('states', 0, 's'),
                ('controls', 0, 'x'),
                ('states', 1, 'S'),
                ('controls', 1, 'X'),
                ('parameters', 0, 'p')
            ]

        }
    }

}

class SymbolicModel:

    def __init__(self, model_name, model_type, symbols, symbolic_equations, symbolic_calibration, symbolic_covariances, symbolic_markov_chain):
        
        self.name = model_name
        self.model_type = model_type
        self.symbols = symbols
        self.equations = symbolic_equations
        self.calibration_dict = symbolic_calibration
        self.covariances = symbolic_covariances
        self.markov_chain = symbolic_markov_chain

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

            raise Exception( "No rule to check model type {}".format(self.model_type))

import re
regex = re.compile("(.*)<=(.*)<=(.*)")
    
def decode_complementarity(comp, control):

    '''
    # comp can be either:
    - None
    - "a<=expr" where a is a controls
    - "expr<=a" where a is a control
    - "expr1<=a<=expr2"
    '''

    try:
        res = regex.match(comp).groups()
    except:
        raise Exception("Unable to parse complementarity condition '{}'".format(comp))
    res = [r.strip() for r in res]
    if res[1] != control:
        msg = "Complementarity condition '{}' incorrect. Expected {} instead of {}.".format(comp, control, res[1])
        raise Exception(msg)
    return [res[0], res[2]]


class NumericModel:

    def __init__(self, symbolic_model, options=None, infos=None):

        self.symbolic = symbolic_model
        self.symbols = symbolic_model.symbols

        self.options = options if options is not None else {}

        self.infos = infos if infos is not None else {}

        self.infos['data_layout'] = 'columns'

        self.name = self.infos['name']
        self.model_type = self.infos['type']

        self.__update_calibration__()
        self.__compile_functions__()


    def __update_calibration__(self):

        # updates calibration according to the symbolic definitions

        system = self.symbolic.calibration_dict 

        from triangular_solver import solve_triangular_system

        self.calibration_dict = solve_triangular_system( system )

        from misc2 import calibration_to_vector
        self.calibration = calibration_to_vector(self.symbols, self.calibration_dict)


        if self.symbolic.covariances is not None:

            import numpy
            covs = numpy.zeros_like(self.symbolic.covariances, dtype=float)
            for i in range(covs.shape[0]):
                for j in range(covs.shape[1]):
                    covs[i,j] = eval(str(self.symbolic.covariances[i,j]), self.calibration_dict )

            self.covariances = covs



    def get_calibration(self, pname, *args):

        if isinstance(pname, list):
            return [ self.get_calibration(p) for p in pname ]
        elif isinstance(pname, tuple):
            return tuple( [ self.get_calibration(p) for p in pname ] )
        elif len(args)>0:
            pnames = (pname,) + args
            return self.get_calibration(pnames)

        group = [g for g in self.symbols.keys() if pname in self.symbols[g]]
        try:
            group = group[0]
        except:
            raise Exception('Unknown symbol {}.')
        i = self.symbols[group].index(pname)
        v = self.calibration[group][i]

        return v


    def set_calibration(self, *args, **kwargs):

        # raise exception if unknown symbol ?

        if len(args)==2:
            pname, pvalue = args
            if isinstance(pname, str):
                self.set_calibration(**{pname:pvalue})
        else:
            # else ignore pname and pvalue
            calib =  self.symbolic.calibration_dict
            calib.update(kwargs)
            self.__update_calibration__()
    
    def __str__(self):
        
        s = '''Model object:
- name: "{name}"
- type: "{type}"
- file: "{filename}\n'''.format(**self.infos)
        
        # import pprint
        # s += '- residuals:\n'
        # s += pprint.pformat(compute_residuals(self),indent=2, depth=1)

        return s

    @property
    def x_bounds(self):

        if 'arbitrage_ub' in self.functions:
            fun_lb = self.functions['arbitrage_lb']
            fun_ub = self.functions['arbitrage_ub']
            return [fun_lb, fun_ub]
        else:
            return None



    def __compile_functions__(self):

        from dolo.compiler.function_compiler_ast import compile_function_ast, eval_ast
        from misc2 import allocating_function
        from dolo.compiler.function_compiler import vector_or_matrix, standard_function

        # works for fg models only
        recipe = recipes['fg']

        comps = []

        functions = {}
        for funname in 'transition','arbitrage':

            if funname == 'transition':

                eqs = self.symbolic.equations[funname]
                eqs = [eq.split('=')[1] for eq in eqs]
                eqs = [str.strip(eq) for eq in eqs]

                n_output = len(self.symbols['states'])

            elif funname == 'arbitrage':

                eqs = []

                for i,eq in enumerate(self.symbolic.equations[funname]):

                    if '|' in eq:
                        control = self.symbols['controls'][i]
                        eq, comp = str.split(eq,'|')
                        lhs, rhs = decode_complementarity(comp, control)
                        comps.append([lhs, rhs])
                    else:
                        comps.append(['-inf', 'inf'])

                    if '=' in eq:
                        lhs, rhs = str.split(eq,'=')
                        eq = '{} - ( {} )'.format(rhs, lhs)

                    eq = str.strip(eq)
                    eqs.append(eq)

                n_output = len(self.symbols['controls'])


            symbols = self.symbols # should match self.symbols
            arg_names = recipe['specs'][funname] # should match self.symbols
            fun = compile_function_ast(eqs, symbols, arg_names, funname=funname)

            functions[funname] = standard_function(fun, n_output )
            

        arg_names = [('states', 0, 's'), ('parameters', 0, 'p')]

        lower_bound = compile_function_ast([c[0] for c in comps], symbols, arg_names, funname='arbitrage_lb')
        upper_bound = compile_function_ast([c[1] for c in comps], symbols, arg_names, funname='arbitrage_ub')
        n_output = len(self.symbols['controls'])
        
        functions['arbitrage_ub'] = standard_function(upper_bound, n_output )
        functions['arbitrage_lb'] = standard_function(lower_bound, n_output )


        self.functions = functions

def yaml_import(fname):

    symbol_types = ['states', 'controls', 'shocks', 'parameters']


    with open(fname) as f:
        txt = f.read()

    txt = txt.replace('^','**')

    import yaml
    
    try:
        data = yaml.safe_load(txt)
    except Exception as e:
        raise e


    if not 'model_type' in data:
        raise Exception("Missing key: 'model_type'.")
    else:
        model_type = data['model_type']

    # if model_type == 'fga':
    #     raise Exception("Model type 'fga' is deprecated. Replace it with 'fg'.")



    if not 'name' in data:
        raise Exception("Missing key: 'name'.")

    if not 'symbols' in data:
        raise Exception("Missing section: 'symbols'.")
    
    # check equations
    if not 'equations' in data:
        raise Exception("Missing section: 'equations'.")


    if not 'calibration' in data:
        raise Exception("Missing section: 'calibration'.")



    # model specific

    if model_type in ('fga','fgh','vfi'):
        if not 'covariances' in data:
            raise Exception("Missing section (model {}): 'covariances'.".format(model_type))
        symbolic_covariances = data['covariances']

    if model_type in ('mfg','mvfi'):
        if not 'markov_chain' in data:
            raise Exception("Missing section (model {}): 'markov_chain'.".format(model_type))
        symbolic_markov_chain = data['markov_chain']



    model_name = data['name']
    symbols = data['symbols']
    symbolic_equations = data['equations']
    symbolic_calibration = data['calibration']

    # shocks are initialized to zero if not calibrated
    initial_values = {
        'shocks': 0,
        'markov_states': 0,
        'controls': float('nan'),
        'states': float('nan')
    }

    for symbol_group,default in initial_values.iteritems():
        if symbol_group in symbols:
            for s in symbols[symbol_group]:
                if s not in symbolic_calibration:
                    symbolic_calibration[s] = default


    # read covariance matrix
    import numpy
    symbolic_covariances = data.get('covariances')
    if symbolic_covariances is not None:
        try:
            tl = numpy.array(symbolic_covariances, dtype='object')
        except:
            msg = "Impossible to read covariances matrix from: {}.".format(symbolic_covariances)
            raise Exception( msg )
        try:
            assert( tl.ndim == 2 )
            assert( tl.shape[0] == tl.shape[1] )
        except:
            msg = "Covariances matrix should be square. Found a {} matrix".format(tl.shape)
            raise Exception(msg)
        symbolic_covariances = tl



    symbolic_markov_chain = data.get('markov_chain')
    # TODO: read markov chain


    options = data.get('options')

    infos = dict()
    infos['filename'] = fname
    infos['name'] = model_name
    infos['type'] = model_type

    smodel = SymbolicModel(model_name, model_type, symbols, symbolic_equations, symbolic_calibration, symbolic_covariances, symbolic_markov_chain)

    model = NumericModel(smodel, infos=infos, options = options)

    return model

if __name__ == "__main__":

    fname = "../../examples/global_models/rbc_temp.yaml"

    model = yaml_import(fname)

    print("calib")
    # print(model.calibration['parameters'])

    print(model.get_calibration(['beta','rk']))
    model.set_calibration(beta=0.95)

    print( model.get_calibration(['beta','rk']))


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