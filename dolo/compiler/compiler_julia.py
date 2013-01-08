from dolo.symbolic.symbolic import TSymbol, Parameter

recipe = dict(

    variable_type = ['states', 'controls', 'auxiliary'],

    equation_type = dict(

        arbitrage = [
            ('states',0),
            ('controls',0),
            ('auxiliary',0),
            ('states',1),
            ('controls',1),
            ('auxiliary',1)
        ],

        transition = [
            ('states',0),
            ('states',-1),
            ('controls',-1),
            ('auxiliary',-1),
            ('shocks',0)
        ],

        auxiliary = [
            ('auxiliary',0),
            ('states',0),
            ('controls',0)
        ]

    )
)

def validate(model, recipe):

    for vg in model['variables_groups']:
        assert( vg in recipe['variable_type'])

    for eqg in model['equations_groups']:
        assert( eqg in recipe['equation_type'])

        allowed_symbols = []

        for syms in recipe['equation_type'][eqg]:
            [sgn,time] = syms
            if syms[0] == 'shocks':
                allowed_symbols += [ s(time) for s in model['shocks_ordering'] ]
            else:
                allowed_symbols += [ s(time) for s in model['variables_groups'][sgn] ]

        allowed_symbols += model['parameters_ordering']

        for eq in model['equations_groups'][eqg]:

            for a in eq.atoms():
                if isinstance(a, (TSymbol, Parameter)):

                    if not a in allowed_symbols:

                        raise Exception('Unexpected symbol {0} in equation \n{1}'.format(a, eq))


class CompilerJulia:

    def __init__(self, model):

        self.model = model

    def process_output(self):

        parms = model['parameters_ordering']

        fun_text = ''

        for eqg in self.model['equations_groups']:

            args = []

            arg_names = []
            for syms in recipe['equation_type'][eqg]:
                [sgn,time] = syms
                if syms[0] == 'shocks':
                    args.append( [ s(time) for s in model['shocks_ordering'] ] )
                else:
                    args.append( [ s(time) for s in model['variables_groups'][sgn] ] )
                if time == 1:
                    stime = '_f'
                elif time == -1:
                    stime = '_p'
                else:
                    stime = ''
                arg_names.append( sgn + stime)

            equations = self.model['equations_groups'][eqg]
            from dolo.compiler.function_compiler_julia import compile_multiargument_function

            txt = compile_multiargument_function(equations, args, arg_names, parms, fname = eqg)

            fun_text += txt

        [y,x,p] = model.read_calibration()
        s_ss = [y[model.variables.index(v)] for v in model['variables_groups']['states'] ]
        x_ss = [y[model.variables.index(v)] for v in model['variables_groups']['controls'] ]
        a_ss = [y[model.variables.index(v)] for v in model['variables_groups']['controls'] ]


        full_text = '''
        {function_definitions}

        model = Model(
            [{states}],
            [{controls}],
            [{auxiliaries}],
            [{parameters}],
            [{shocks}],
            transition,
            arbitrage,
            auxiliary
            {params},
            {s_ss},
            {x_ss},
            {a_ss}
        )

        '''.format(
            function_definitions = fun_text,
            states = str.join(',', ["'{}'".format(e ) for e in model['variables_groups']['states']]),
            controls = str.join(',', ["'{}'".format(e ) for e in model['variables_groups']['controls']]),
            auxiliaries = str.join(',', ["'{}'".format(e ) for e in model['variables_groups']['auxiliary']]),
            parameters = str.join(',', ["'{}'".format(e ) for e in model['parameters_ordering']]),
            shocks = str.join(',', ["'{}'".format(e ) for e in model['shocks_ordering']]),
            params = str(p),
            s_ss = str(s_ss),
            x_ss = str(x_ss),
            a_ss = str(a_ss)

        )

        print(full_text)

if __name__ == '__main__':

    from dolo import *

    model = yaml_import('examples/global_models/rbc.yaml')

    validate(model, recipe)

    comp = CompilerJulia(model)

    comp.process_output()





