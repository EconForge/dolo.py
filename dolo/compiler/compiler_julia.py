
class CompilerJulia:

    def __init__(self, model, model_type=None):

        if model_type is None:
            model_type = model['original_data']['model_type']


        self.model = model

        from dolo.symbolic.validator import validate
        if isinstance(model_type, str):
            validate(model, model_type)
            from dolo.symbolic.recipes import recipes
            self.recipe = recipes[model_type]
        else:
            # if model_type is not a string, we assume it is a recipe
            self.recipe = model_type


    def process_output(self, recipe=None):

        recipe = self.recipe

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

        calib = model.calibration

        steady_state = calib['steady_state']
        parameters_values = calib['parameters']


        funs_text = "functions = {\n"
        for fun_name in recipe['equation_type']:
            funs_text += '\t"{0}" => {0},\n'.format(fun_name)
        funs_text += '}'

        ss_text = "steady_state = {\n"
        for k,v in steady_state.iteritems():
            ss_text += '\t"{0}" => {1},\n'.format( k, str(v) )
        ss_text += '}'

        var_text = "variables = {\n"
        for vn, vg in model['variables_groups'].iteritems():
            var_text += '\t"{0}" => [{1}],\n'.format(vn, str.join(',', ['"{}"'.format(e ) for e in vg]))

        var_text += '\t"parameters" => [{}],\n'.format(str.join(',', ['"{}"'.format(e ) for e in model['parameters_ordering']]))
        var_text += '\t"shocks" => [{}]\n'.format(str.join(',', ['"{}"'.format(e ) for e in model['shocks_ordering']]))

        var_text += '}'

        full_text = '''
{function_definitions}

{funs_text}

{ss_text}

{var_text}


calibration = {{
    "steady_state" => steady_state,
    "parameters" => {params}
}}

model = {{
    "variables" => variables,
    "functions" => functions,
    "calibration" => calibration
}}
'''.format(
            function_definitions = fun_text,
            funs_text = funs_text,
            ss_text = ss_text,
            var_text = var_text,
            params = str(parameters_values)

        )

        return full_text

if __name__ == '__main__':

    from dolo import *

    model = yaml_import('examples/global_models/rbc.yaml')
    comp = CompilerJulia(model)

    print  comp.process_output()

    print("******10")
    print("******10")
    print("******10")

    model = yaml_import('examples/global_models/optimal_growth.yaml')
    comp = CompilerJulia(model)

    print  comp.process_output()





