from dolo.symbolic.symbolic import TSymbol, Parameter


from dolo.symbolic.recipes import recipes

def validate(model, recipe):

#    if isinstance(recipe, str):
#        validate(model, recipes[recipe])
#        return

    for vg in model.symbols_s:
        if not vg in ('parameters','shocks'):
            assert( vg in recipe['variable_type'])

    for eqg in model.equations_groups:
        assert( eqg in recipe['equation_type'])

        eq_recipe = recipe['equation_type'][eqg]

        if isinstance(eq_recipe, list):
            eq_recipe = {'definition': False, 'lhs': recipe['equation_type'][eqg], 'rhs': recipe['equation_type'][eqg]}

        if eq_recipe['definition']:
            lhs_symbols = tuple( [eq.lhs for eq in model.equations_groups[eqg]] )
            lhs_type =  eq_recipe['lhs'][0][0]
            correct_symbols = tuple( model.symbols_s[lhs_type] )
            if lhs_symbols != correct_symbols:
                raise(Exception('''
    Blocks of type "{0}" must list variables of type "{1}" in the declaration order.
    Declaration order : {2}
    Definition order : {3}'''.format(eqg, lhs_type, correct_symbols, lhs_symbols)))

        equations = model.equations_groups[eqg]

        if eq_recipe.get('definition'):
            from dolo.compiler.common import solve_recursive_block
            try:
                temp = solve_recursive_block(equations)
            except:
                raise Exception('\n   The equation group "{}" cannot be solved recursively.'.format(eqg) )

        for side in ('lhs', 'rhs'):
            allowed_symbols = []

            for syms in eq_recipe[side]:

                [sgn,time] = syms
                if syms[0] == 'shocks':
                    allowed_symbols += [ s(time) for s in model.symbols_s['shocks'] ]
                else:
                    allowed_symbols += [ s(time) for s in model.symbols_s[sgn] ]

            if eq_recipe.get('definition') and side == 'rhs':
                allowed_symbols += [ eq.lhs for eq in equations ]

            # by default recursive blocs are allowed

            allowed_symbols += model.symbols_s['parameters']

            allowed_symbols += model.symbols_s['parameters']


            for eq in equations:
                expr = eq.lhs if side=='lhs' else eq.rhs
                for a in expr.atoms():
                    if isinstance(a, (TSymbol, Parameter)):
                        if not a in allowed_symbols:
                            raise Exception('Unexpected symbol {0} in equation \n{1}'.format(a, eq))


if __name__ == '__main__':
    from dolo import *
    model = yaml_import('examples/global_models/rbc.yaml')
    from dolo.symbolic.recipes import recipe_fga
    validate(model, recipe_fga)
