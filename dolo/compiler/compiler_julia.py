
class CompilerJulia:

    def __init__(self, model):

        self.model = model

        from dolo.symbolic.validator import validate
        validate(model,'fga')

    def process_output(self, recipe=None):

        if recipe is None:
            from dolo.symbolic.recipes import recipe_fga as recipe

        model = self.model
        parms = model['parameters_ordering']


        fun_text = ''

        for eqg in self.model['equations_groups']:

            args = []

            is_a_definition = 'definition' in recipe['equation_type'][eqg]

            if is_a_definition:
                arg_specs = recipe['equation_type'][eqg]['rhs']
            else:
                arg_specs = recipe['equation_type'][eqg]


            arg_names = []
            for syms in arg_specs:
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

            if is_a_definition:
                equations = [eq.rhs for eq in equations]
            else:
                equations = [eq.gap for eq in equations]

            from dolo.compiler.function_compiler_julia import compile_multiargument_function

            txt = compile_multiargument_function(equations, args, arg_names, parms, fname = eqg)

            fun_text += txt


        # the following part only makes sense for fga models

        [y,x,p] = model.read_calibration(to_numpy=False)

        s_ss = [y[model.variables.index(v)] for v in model['variables_groups']['states'] ]
        x_ss = [y[model.variables.index(v)] for v in model['variables_groups']['controls'] ]
        a_ss = [y[model.variables.index(v)] for v in model['variables_groups']['controls'] ]


        full_text = '''
{function_definitions}

model = {{
    "states" => [{states}],
    "controls" => [{controls}],
    "auxiliaries" => [{auxiliaries}],
    "parameters" => [{parameters}],
    "shocks" => [{shocks}],
    "transition" => transition,
    "arbitrage" => arbitrage,
    "auxiliary" => auxiliary,
    "params" => {params},
    "s_ss" => {s_ss},
    "x_ss" => {x_ss},
    "a_ss" => {a_ss}
}}
'''.format(
            function_definitions = fun_text,
            states = str.join(',', ['"{}"'.format(e ) for e in model['variables_groups']['states']]),
            controls = str.join(',', ['"{}"'.format(e ) for e in model['variables_groups']['controls']]),
            auxiliaries = str.join(',', ['"{}"'.format(e ) for e in model['variables_groups']['auxiliary']]),
            parameters = str.join(',', ['"{}"'.format(e ) for e in model['parameters_ordering']]),
            shocks = str.join(',', ['"{}"'.format(e ) for e in model['shocks_ordering']]),
            params = str(p),
            s_ss = str(s_ss),
            x_ss = str(x_ss),
            a_ss = str(a_ss)

        )

        return full_text

if __name__ == '__main__':

    from dolo import *

    model = yaml_import('examples/global_models/rbc.yaml')

    comp = CompilerJulia(model)

    print  comp.process_output()





