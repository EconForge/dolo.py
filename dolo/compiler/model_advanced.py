raise Exception("Not supported.")

class Model:

    def __init__(self, symbols, calibration_dict, funs, options=None, infos=None):

        self.symbols = symbols

        self.source = dict()
        self.source['functions'] = funs
        self.source['calibration'] = calibration_dict


        [calibration, functions] = prepare_model(symbols, calibration_dict, funs)
        self.functions = functions
        self.calibration = calibration


        self.options = options if options is not None else {}
        self.infos = infos if infos is not None else {}

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
            calib =  self.source['calibration']
            calib.update(kwargs)
            self.calibration = read_calibration(self.symbols, calib)

    def __str__(self):

        s = '''Model object:
- name: "{name}"
- type: "{type}"
- file: "{filename}\n'''.format(**self.infos)

        import pprint
        s += '- residuals:\n'
        s += pprint.pformat(compute_residuals(self),indent=2, depth=1)

        return s





def prepare_model(symbols, calibration_dict, funs):

    import inspect

    calibration = read_calibration(symbols, calibration_dict)
    functions = dict()

#    for k,f in funs.items():
    for k,f in funs.items():

        argspec = inspect.getargspec(f)
        if k == 'transition':
            size_output = len( symbols['states'] )
            k = 'transition'
        elif k == 'arbitrage':
            size_output = len( symbols['controls'] )
            k = 'arbitrage'
        else:
            continue

        functions[k] = allocating_function( f, size_output )

    return [calibration,functions]

def import_model(filename):

#    d = {}
#    e = {}
    d = {}
    e = {}
    with open(filename) as f:
        code = compile(f.read(), filename, "exec")

    # TODO improve message
    exec(code, e, e)

    symbols = e['symbols']
    calibration_dict = e['calibration_dict']
    funs = {
            "transition": e["transition"],
            "arbitrage": e["arbitrage"],
            "markov_chain": e["markov_chain"],
        }
    if "complementarities" in e:
        funs[ "complementarities"] = e["complementarities"]

    model_spec = e['model_spec']
    model_name = e.get('name')
    if model_name is None:
        model_name = 'anonymous'

    infos = dict()
    infos['filename'] = filename
    infos['type'] = model_spec
    infos['name'] = model_name

    if 'options' in e:
        options = e['options']
    else:
        options = {}


    model = Model(symbols, calibration_dict, funs, options, infos)

    return model