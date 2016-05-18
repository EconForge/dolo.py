# all symbolic functions take ast expression trees (not expressions) as input
# this one can be constructed as : ast.parse(s).body[0].value

import ast

from ast import UnaryOp, UAdd, USub, Name, Load, Call
from ast import NodeTransformer


def std_tsymbol(tsymbol):
    s, date = tsymbol
    if date == 0:
        return '_{}_'.format(s)
    elif date <= 0:
        return '_{}_m{}_'.format(s, str(-date))
    elif date >= 0:
        return '_{}__{}_'.format(s, str(date))

class StandardizeDatesSimple(NodeTransformer):

    # replaces calls to variables by time subscripts

    def __init__(self, variables):

        self.variables = variables
        # self.variables = tvariables # ???

    def visit_Name(self, node):

        name = node.id
        if name in self.variables:
            return Name(id=std_tsymbol((name,0)),ctx=Load())
        else:
            return node

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            newname = std_tsymbol((name, date))
            if newname is not None:
                return Name(newname, Load())

        else:

            # , keywords=node.keywords, starargs=node.starargs, kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])



class TimeShiftTransformer(ast.NodeTransformer):
    def __init__(self, variables, shift=0):

        self.variables = variables
        self.shift = shift

    def visit_Name(self, node):
        name = node.id
        if name in self.variables:
            if self.shift==0 or self.shift=='S':
                return ast.parse(name).body[0].value
            else:
                return ast.parse('{}({})'.format(name,self.shift)).body[0].value
        else:
             return node

    def visit_Call(self, node):

        name = node.func.id
        args = node.args[0]

        if name in self.variables:
            if isinstance(args, UnaryOp):
                # we have s(+1)
                if (isinstance(args.op, UAdd)):
                    args = args.operand
                    date = args.n
                elif (isinstance(args.op, USub)):
                    args = args.operand
                    date = -args.n
                else:
                    raise Exception("Unrecognized subscript.")
            else:
                date = args.n
            if self.shift =='S':
                return ast.parse('{}'.format(name)).body[0].value
            else:
                new_date = date+self.shift
                if new_date != 0:
                    return ast.parse('{}({})'.format(name,new_date)).body[0].value
                else:
                    return ast.parse('{}'.format(name)).body[0].value
        else:

            # , keywords=node.keywords,  kwargs=node.kwargs)
            return Call(func=node.func, args=[self.visit(e) for e in node.args], keywords=[])




import copy


def timeshift(expr, variables, shift):

    eexpr = copy.deepcopy(expr)
    return TimeShiftTransformer(variables, shift).visit(eexpr)


class Compare:

    def __init__(self):
        self.d = {}

    def compare(self, A, B):
        if isinstance(A, ast.Name) and (A.id[0] == '_'):
            if A.id not in self.d:
                self.d[A.id] = B
                return True
            else:
                return self.compare(self.d[A.id], B)
        if not (A.__class__ == B.__class__): return False
        if isinstance(A, ast.Name):
            return A.id == B.id
        elif isinstance(A, ast.Call):
            if not self.compare(A.func, B.func): return False
            if not len(A.args)==len(B.args): return False
            for i in range(len(A.args)):
                if not self.compare(A.args[i], B.args[i]): return False
            return True
        elif isinstance(A, ast.Num):
            return A.n == B.n
        elif isinstance(A, ast.Expr):
            return self.compare(A.value, B.value)
        elif isinstance(A, ast.Module):
            if not len(A.body)==len(B.body): return False
            for i in range(len(A.body)):
                if not self.compare(A.body[i], B.body[i]): return False
            return True
        elif isinstance(A, ast.BinOp):
            if not isinstance(A.op, B.op.__class__): return False
            if not self.compare(A.left, B.left): return False
            if not self.compare(A.right, B.right): return False
            return True
        elif isinstance(A, ast.UnaryOp):
            if not isinstance(A.op, B.op.__class__): return False
            return self.compare(A.operand, B.operand)
        elif isinstance(A, ast.Subscript):
            if not self.compare(A.value, B.value): return False
            return self.compare(A.slice, B.slice)
        elif isinstance(A, ast.Index):
            return self.compare(A.value, B.value)
        elif isinstance(A, ast.Compare):
            if not self.compare(A.left, B.left): return False
            if not len(A.ops)==len(B.ops): return False
            for i in range(len(A.ops)):
                if not self.compare(A.ops[i], B.ops[i]): return False
            if not len(A.comparators)==len(B.comparators): return False
            for i in range(len(A.comparators)):
                if not self.compare(A.comparators[i], B.comparators[i]): return False
            return True
        elif isinstance(A, ast.In):
            return True
        elif isinstance(A, (ast.Eq, ast.LtE)):
            return True
        else:
            print(A.__class__)
            raise Exception("Not implemented")


def compare(a,b):
    comp = Compare()
    val = comp.compare(a,b)
    d = comp.d
    return val

def match(m,s):
    comp = Compare()
    val = comp.compare(m,s)
    d = comp.d
    if len(d) == 0:
        return val
    else:
        return d

known_functions = ['log','exp','sin','cos']

import ast

class ListNames(ast.NodeVisitor):
    def __init__(self):
        self.found = []
    def visit_Name(self, name):
        self.found.append(name.id)

def get_names(expr):
    ln = ListNames()
    ln.visit(expr)
    return [e for e in ln.found]

def eval_scalar(tree):
    try:
        if isinstance(tree, ast.Num):
            return tree.n
        elif isinstance(tree, ast.UnaryOp):
            if isinstance(tree.op, ast.USub):
                return -tree.operand.n
        else:
            raise Exception("Don't know how to do that.")
    except:
        raise Exception("Don't know how to do that.")


class ExpressionChecker(ast.NodeVisitor):

    def __init__(self, spec_variables, known_functions, known_constants):
        self.spec_variables = spec_variables
        self.known_functions = known_functions
        self.known_constants = known_constants
        self.functions = []
        self.variables = []
        self.problems = []

    def visit_Call(self, call):
        name = call.func.id
        colno = call.func.col_offset
        if name in self.spec_variables:
            try:
                assert(len(call.args)==1)
                n = eval_scalar(call.args[0])
                allowed_timing = self.spec_variables[name]
                if allowed_timing is None or (n in allowed_timing):
                    self.variables.append((name, n, call.func.col_offset))
                else:
                    self.problems.append([name,n,colno,'incorrect_timing',allowed_timing])
            except Exception as e:
                print(e)
                self.problems.append([name,None,colno,'timing_error'])

        elif name in self.known_functions:
            self.functions.append((name, colno))
            for e in call.args:
                self.visit(e)
        else:
            self.problems.append([name, None, colno,'unknown_function'])

    def visit_Name(self, name):
        # colno = name.colno
        colno = name.col_offset
        n = 0
        name = name.id
        if name in self.spec_variables:
            allowed_timing = self.spec_variables[name]
            if (allowed_timing is None) or (n in allowed_timing):
                self.variables.append((name, n, colno))
            else:
                self.problems.append([name,n,colno,'incorrect_timing',allowed_timing])
        elif name not in self.known_constants:
            self.problems.append([name,0,colno,'unknown_variable'])

def check_expression(expr, spec_variables, known_functions=[]):

    from dolo.compiler.language import functions, constants
    func = list(functions.keys()) + known_functions

    ch = ExpressionChecker(spec_variables, func, constants)
    ch.visit(expr)
    return dict(
        functions = ch.functions,
        variables = ch.variables,
        problems = ch.problems
    )

# def get_variables(variables, expr):
#     ln = ListVariables(variables)
#     ln.visit(expr)
#     return ln.found
#
#
# def get_functions(variables, expr):
#     ln = ListVariables(variables)
#     ln.visit(expr)
#     return ln.functions


#
# class ExpressionChecker(ast.NodeVisitor):
#
#     def __init__(self, variables, functions):
#
#         self.allowed_variables = variables
#         self.functions = functions
#         self.found = []
#         self.problems = []
#
#     def visit_Call(self, call):
#         name = call.func.id
#         if name in self.variables:
#             assert(len(call.args)==1)
#             print(call.args[0])
#             n = eval_scalar(call.args[0])
#             self.found.append((name, n))
#         elif name in self.functions:
#             self.functions.append(name)
#             for e in call.args:
#                 self.visit(e)
#         else:
#             for e in call.args:
#                 self.visit(e)
#
#     def visit_Name(self, name):
#         name = name.id
#         if name in self.variables:
#             self.found.append((name,0))
#
# def check_expression(expr, variables, functions):
#     ec = ExpressionChecker(variables, functions)
#     pbs = ec.visit(ec)
