from dolo.compiler.symbolic import *

def parse(s):
    import ast
    return ast.parse(s).body[0].value

assert( compare(parse('b(_x)'),parse('b(1)')) )

symbols = dict(
    states = ['x','y'],
    parameters = ['a', 'b']
)
spec = [('states',0), ('states',1), ('parameters',0)]


known_functions = ['log', 'exp', 'sin', 'cos']


expression = parse('sin(x(-1) + x + x(1) + y(0)*b + z + y(1-1)) + s')

resp = check_expression(expression, {'x':(0,-1,1),'y':(1,),'b':(0,),'s':None}, ['sin'])
print(resp)

# expression = parse('x(-1) + y*b')
