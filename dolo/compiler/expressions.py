from dolo.compiler.symbolic import eval_scalar
import ast

def parse(s): return ast.parse(s).body[0].value

class ExprVisitor(ast.NodeVisitor):

    def __init__(self, variables):
        self.variables = variables

    def visit_Call(self, call):
        name = call.func.id
        if name in self.variables:
            assert(len(call.args) == 1)
            n = eval_scalar(call.args[0])
            return self.visit_Variable((name, n))
        else:
            return self.visit_RCall(call)

    def visit_RCall(self, call):
        return self.generic_visit(call)

    def visit_Name(self, cname):
        name = cname.id
        if name in self.variables:
            return self.visit_Variable((name, 0))
        else:
            return self.visit_RName(cname)

    def visit_RName(self, name):
        return self.generic_visit(name)

class ExprTransformer(ast.NodeTransformer):

    def __init__(self, variables):
        self.variables = variables

    def visit_Call(self, call):
        name = call.func.id
        if name in self.variables:
            assert(len(call.args) == 1)
            n = eval_scalar(call.args[0])
            return self.visit_Variable((name, n))
        else:
            return self.generic_visit(call)

    def visit_Name(self, cname):
        name = cname.id
        if name in self.variables:
            return self.visit_Variable((name, 0))
        else:
            return self.generic_visit(cname)


class TimeShift(ExprVisitor):

    def __init__(self, variables, shift):
        self.variables = variables
        self.shift = shift

    def visit_Variable(self, tvar):
        name, t = tvar
        return parse( "{}({})".format(name,t+self.shift))


class Apply(ExprVisitor):

    def __init__(self, variables, fun):
        self.variables = variables
        self.fun = fun

    def visit_Variable(self, tvar):
        return self.fun(tvar)

# pp = Apply(['b']).visit(expr)
#
# expr = parse('a+b(1)+c')
#
# pp = TimeShift(['b'],-1).visit(expr)
# print(pp)
#
# from dolo.compiler.codegen import to_source
#
# print(ast.dump(pp))
#
# to_source(pp)
