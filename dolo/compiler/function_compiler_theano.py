from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion
from theano import __version__

theano_less_than_6 = (LooseVersion(__version__) < LooseVersion('0.6.0') )


def compile_multiargument_function(values, args_list, args_names, parms, fname='anonymous_function', diff=True, vectorize=True, return_function=True, order='rows'):
    if order != 'rows':
        raise Exception('Only row order implemented')

    source = compile_theano_source(values, args_list, args_names, parms, fname=fname)
    exec(source)
    return f


def compile_theano_source(values, args_list, args_names, parms, fname='anonymous_function'):


    """
    :param vars: parameters
    :param values: list of values (already triangular)


    """


    vars = ['_res_{}'.format(i) for i in range(len(values))]

    sub_dict = {}
    for e in vars:
        try:
            sn = e.safe_name()
        except Exception:
            sn = '_'+str(e)
        sub_dict[e] = sn

    from dolo.compiler.common import DicPrinter

    dec = ''

    for s in args_names:
        dec += "{} = T.matrix('{}')\n".format(s,s)

    dec += "p = T.vector('p')\n"

    for i,p in enumerate(parms):
        sn = '_'+str(p)
        sub_dict[p] = sn
        dec += '{} = p[{}]\n'.format(sn,i)

    for i, l in enumerate( args_list):
        name = args_names[i]
        for j, e in enumerate(l):
            try:
                sn = e.safe_name()
            except Exception:
                sn = '_'+str(e)
            sub_dict[e] = sn
            dec += '{} = {}[{},:]\n'.format(sn,name,j)

    dp = DicPrinter(sub_dict)
    strings = [ ]
    for i,eq in enumerate( values ):
        rhs = ( dp.doprint( eq) )
        lhs = vars[i]
#        strings.append( '{} = OO + {}'.format(lhs,rhs))
        strings.append( '{} = {}'.format(lhs,rhs))

    source = """

from theano import tensor as T
from theano import function
from theano.tensor import exp

{declarations}

#    OO =  T.zeros((1,s.shape[1]))

{computations}

res = T.stack({vars})

f = function([{args}], res, mode='FAST_RUN',name="{fname}"{on_unused_input})

"""

#    print(args_names)
    source = source.format(
        computations = str.join( '\n', strings),
        declarations = dec,
        vars = str.join(', ', [str(v) for v in vars]),
        args = str.join(', ', [str(v) for v in args_names] + ['p'] ),
        fname = fname,
        on_unused_input = ",on_unused_input='ignore'" if not theano_less_than_6 else ""

    )

    return source
