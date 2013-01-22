from dolo.symbolic.symbolic import TSymbol, Parameter


from dolo.symbolic.recipes import recipes

def validate(model, recipe):

#    if isinstance(recipe, str):
#        validate(model, recipes[recipe])
#        return

    for vg in model['variables_groups']:
        assert( vg in recipe['variable_type'])

    for eqg in model['equations_groups']:
        assert( eqg in recipe['equation_type'])

        eq_recipe = recipe['equation_type'][eqg]

        if isinstance(eq_recipe, list):
            eq_recipe = {'definition': False, 'lhs': recipe['equation_type'][eqg], 'rhs': recipe['equation_type'][eqg]}

        for side in ('lhs', 'rhs'):
            allowed_symbols = []

            for syms in eq_recipe[side]:

                [sgn,time] = syms
                if syms[0] == 'shocks':
                    allowed_symbols += [ s(time) for s in model['shocks_ordering'] ]
                else:
                    allowed_symbols += [ s(time) for s in model['variables_groups'][sgn] ]

            allowed_symbols += model['parameters_ordering']

            for eq in model['equations_groups'][eqg]:
                expr = eq.lhs if side=='lhs' else eq.rhs
                for a in expr.atoms():
                    if isinstance(a, (TSymbol, Parameter)):

                        if not a in allowed_symbols:

                            raise Exception('Unexpected symbol {0} in equation \n{1}'.format(a, eq))
