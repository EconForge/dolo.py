'''
ideas :
-  recursive blocks           [by default]
- (order left hand side ?)    [by default]
- dependency across blocks
- dummy blocks that are basically substituted everywhere else
'''




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

#import copy
#recipe_fgae = copy.deepcopy(recipe_fga)
#recipe_fgae['equation_type']['arbitrage'].append( ('shocks',1) )

recipe_fgh1 = dict(

    model_type = 'fgh1',

    variable_type = ['states', 'controls', 'expectations'],

    equation_type = dict(

        arbitrage = [
            ('states',0),
            ('controls',0),
            ('expectations',0),
        ],

        transition = {
            'definition': True,
            'lhs': [
                ('states',0),
                ],
            'rhs': [
                ('states',-1),
                ('controls',-1),
                ('shocks',0)
            ]
        },

        expectation = {

            'definition': True,

            'lhs': [
                ('expectations',0),
                ],

            'rhs': [
                ('states',1),
                ('controls',1)
            ]
        }

    )
)

recipe_fgh2 = dict(

    model_type = 'fgh2',

    variable_type = ['states', 'controls', 'expectations'],

    equation_type = dict(

        arbitrage = [
            ('states',0),
            ('controls',0),
            ('expectations',0),
        ],

        transition = {
            'definition': True,
            'lhs': [
                ('states',0),
                ],
            'rhs': [
                ('states',-1),
                ('controls',-1),
                ('shocks',0)
            ]
        },

        expectation = {

            'definition': True,

            'lhs': [
                ('expectations',0),
                ],

            'rhs': [
                ('states',0),
                ('controls',0),
                ('shocks',1),
                ('states',1),
                ('controls',1)
            ]
        }

    )
)


recipes = {
    'fga':  recipe_fga,
#    'fgae': recipe_fgae,
    'fgh1': recipe_fgh1,
    'fgh2': recipe_fgh2,
    'fgh':  recipe_fgh2,
}
