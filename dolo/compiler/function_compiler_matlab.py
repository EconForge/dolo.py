"""

Symbolic expressions can be evaluated by substituting the values of each symbol. It is however an expensive operation
 which becomes very costly when the number of evaluations grows.

The functions in this module  take a list of symbolic expressions representing a function :math:`R^p \\rightarrow R^n`
and turn it into an efficient python function, which can be evaluated repeatedly at lower cost. They use one of the next
libraries for efficient vectorization: `numpy <http://numpy.scipy.org/>`_, `numexpr <http://code.google.com/p/numexpr/>`_ or `theano <http://deeplearning.net/software/theano/>`_:

"""


from __future__ import division


def compile_multiargument_function(equations, args_list, args_names, parms, diff=False, fname='anonymous_function'):
    """
    :param equations: list of sympy expressions
    :param args_list: list of lists of symbols (e.g. [[a_1,a_2], [b_1,b_2]])
    :param args_names: list of strings (['a','b']
    :param parms: list of symbols to be used as parameters
    :param fname: name of the function to be generated
    :param diff: include symbolic derivatives in generated function
    :param vectorize: arguments are vectorized (not parameters)
    :param return_function: the source of a Matlab function f(a,b,p) where p is a vector of parameters and a, b, arrays
    :return:
    """

    template = '{0}(:,{1})'

    sub_list = {}

    for i,args in enumerate(args_list):
        vec_name = args_names[i]
        for j,v in enumerate(args):
            sub_list[v] = template.format(vec_name,j+1)

    for i,p in enumerate(parms):
        sub_list[p] = '{0}({1})'.format('p',i+1)

    import sympy
    sub_list[sympy.Symbol('inf')] = 'inf'


    text = '''
function [{return_names}] = {fname}({args_names}, {param_names}, output)

    if nargin <= {nargs}
        output = zeros({nargs},1);
        for i = 1:nargout
            output(i) = 1;
        end
    end

    n = size({var},1);

{content}
end
'''

    from dolo.compiler.common import DicPrinter

    dp = DicPrinter(sub_list)

    def write_eqs(eq_l,outname='val'):
        eq_block = '    {0} = zeros( n, {1} );\n'.format(outname, len(eq_l))
        for i,eq in enumerate(eq_l):
            eq_block += '    {0}(:,{1}) = {2};\n'.format(outname, i+1,  dp.doprint_matlab(eq, vectorize=True))
        return eq_block

    def write_der_eqs(eq_l,v_l,lhs):
        '''Format Jacobians'''
        eq_block = '        {lhs} = zeros( n,{0},{1} );\n'.format(len(eq_l),len(v_l),lhs=lhs)
        eq_l_d = eqdiff(eq_l,v_l)
        for i,eqq in enumerate(eq_l_d):
            for j,eq in enumerate(eqq):
                s = dp.doprint_matlab( eq, vectorize=True )
                if s != "0":
                    eq_block += '        {lhs}(:,{0},{1}) = {2};\n'.format(i+1,j+1,s,lhs=lhs)
        return eq_block

    content = write_eqs(equations)

    content += '''
    if nargout <= 1
        return
    end
'''

    if diff:
        for i,a_g in enumerate(args_list):
            content += "\n    % Derivatives w.r.t: {0}\n\n".format(args_names[i])
            lhs = 'val_' + args_names[i]
            content += '    if output({})\n'.format(i+2)
            content += write_der_eqs(equations, a_g, lhs)
            content += '    else\n'
            content += '        val_{} = [];\n'.format(args_names[i])
            content += '    end;\n'

    return_names = str.join(', ', ['val'] + [ 'val_'+ str(a) for a in args_names] ) if diff else 'val'
    text = text.format(
            fname = fname,
            nargs = len(args_names)+1,
            var = args_names[0],
            content = content,
            return_names = return_names,
            args_names = str.join(', ', args_names),
            param_names = 'p'
            )

#    text = text.replace('+','.+')
#    text = text.replace('-','.-')


    return text

def compile_incidence_matrices(equations, args_list):
    '''Calculate the incidence matrices of a system of equations with respect to several sets of variables and convert it to MATLAB cell array'''
    from dolo.misc.matlab import value_to_mat
    text = '''{'''
    for i,a_g in enumerate(args_list):
        text += value_to_mat(JacobianStructure(equations,a_g)).replace('[[','[').replace(']]',']') + ' '
    text += '};'
    return text

def code_to_function(text, name):
    d = {}
    e = {}
    exec(text, d, e)
    return e[name]


def eqdiff(leq,lvars):
    '''Calculate the Jacobian of the system of equations with respect to a set of variables.'''
    from sympy import powsimp
    resp = []
    for eq in leq:
        el = [ powsimp(eq.diff(v)) for v in lvars]
        resp += [el]
    return resp

def JacobianStructure(leq,lvars):
    '''Calculate the incidence matrix of a system of equations with respect to one set of variables'''
    from numpy import array
    jac_struc = array([[0 for i in range(len(leq))] for j in range(len(lvars))])
    for i in range(len(leq)):
        for j in range(len(lvars)):
            if leq[i].diff(lvars[j]) != 0:
                jac_struc[j][i] = 1
    jac_struc = jac_struc.T
    return jac_struc


if __name__ == '__main__':

    import sympy
    from pprint import pprint

    [w,x,y,z,t] = vars = sympy.symbols('w, x, y, z, t')
    [a,b,c,d] = parms = sympy.symbols('a, b, c, d')

    [k_1,k_2] = s_sym = sympy.symbols('k_1, k_2')
    [x_1,x_2] = x_sym = sympy.symbols('x_1, x_2')

    args_list = [
        s_sym,
        x_sym
    ]

    from sympy import exp

    eqs = [
        x + y*k_2 + z*exp(x_1 + t),
        (y + z)**0.3,
        z,
        (k_1 + k_2)**0.3,
        k_2**x_1
    ]

    sdict = {s:eqs[i] for i,s in enumerate(vars) }

    from dolo.misc.triangular_solver import solve_triangular_system
    order = solve_triangular_system(sdict, return_order=True)

    ordered_vars  = [ v for v in order ]
    ordered_eqs = [ eqs[vars.index(v)] for v in order ]

    pprint(ordered_vars)
    pprint(ordered_eqs)



    #    f = create_fun()
    #
    #    test = f(s1,x1,p0)
    #    print(test)
    args_names = ['s','x']
    #
    #
    solution = solve_triangular_system(sdict)
    vals = [sympy.sympify(solution[v]) for v in ordered_vars]


    output = compile_multiargument_function( vals, args_list, args_names, parms, diff=True, fname='test' )

    print(output)

    with file('test.m','w') as f:
        f.write(output)
