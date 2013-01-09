recipe_fga = dict(

    model_type = 'fga',

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

        transition = {
            'definition': True,
            'lhs': [
                ('states',0),
                ],
            'rhs': [
                ('states',-1),
                ('controls',-1),
                ('auxiliary',-1),
                ('shocks',0)
            ]
        },

        auxiliary = {

            'definition': True,

            'lhs': [
                ('auxiliary',0),
                ],

            'rhs': [
                ('states',0),
                ('controls',0)
            ]
        }

    )
)

recipes = {
    'fga': recipe_fga
}