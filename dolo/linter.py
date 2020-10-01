# import ast
# import json
# import ruamel.yaml as ry
# from ruamel.yaml.comments import CommentedSeq
# from dolo.compiler.symbolic import check_expression
# from dolo.compiler.recipes import recipes
# from dolo.misc.termcolor import colored


# class Compare:
#     def __init__(self):
#         self.d = {}

#     def compare(self, A, B):
#         if isinstance(A, ast.Name) and (A.id[0] == '_'):
#             if A.id not in self.d:
#                 self.d[A.id] = B
#                 return True
#             else:
#                 return self.compare(self.d[A.id], B)
#         if not (A.__class__ == B.__class__):
#             return False
#         if isinstance(A, ast.Name):
#             return A.id == B.id
#         elif isinstance(A, ast.Call):
#             if not self.compare(A.func, B.func):
#                 return False
#             if not len(A.args) == len(B.args):
#                 return False
#             for i in range(len(A.args)):
#                 if not self.compare(A.args[i], B.args[i]):
#                     return False
#             return True
#         elif isinstance(A, ast.Num):
#             return A.n == B.n
#         elif isinstance(A, ast.Expr):
#             return self.compare(A.value, B.value)
#         elif isinstance(A, ast.Module):
#             if not len(A.body) == len(B.body):
#                 return False
#             for i in range(len(A.body)):
#                 if not self.compare(A.body[i], B.body[i]):
#                     return False
#             return True
#         elif isinstance(A, ast.BinOp):
#             if not isinstance(A.op, B.op.__class__):
#                 return False
#             if not self.compare(A.left, B.left):
#                 return False
#             if not self.compare(A.right, B.right):
#                 return False
#             return True
#         elif isinstance(A, ast.UnaryOp):
#             if not isinstance(A.op, B.op.__class__):
#                 return False
#             return self.compare(A.operand, B.operand)
#         elif isinstance(A, ast.Subscript):
#             if not self.compare(A.value, B.value):
#                 return False
#             return self.compare(A.slice, B.slice)
#         elif isinstance(A, ast.Index):
#             return self.compare(A.value, B.value)
#         elif isinstance(A, ast.Compare):
#             if not self.compare(A.left, B.left):
#                 return False
#             if not len(A.ops) == len(B.ops):
#                 return False
#             for i in range(len(A.ops)):
#                 if not self.compare(A.ops[i], B.ops[i]):
#                     return False
#             if not len(A.comparators) == len(B.comparators):
#                 return False
#             for i in range(len(A.comparators)):
#                 if not self.compare(A.comparators[i], B.comparators[i]):
#                     return False
#             return True
#         elif isinstance(A, ast.In):
#             return True
#         elif isinstance(A, (ast.Eq, ast.LtE)):
#             return True
#         else:
#             print(A.__class__)
#             raise Exception("Not implemented")


# def compare_strings(a, b):
#     t1 = ast.parse(a)
#     t2 = ast.parse(b)
#     comp = Compare()
#     val = comp.compare(t1, t2)
#     return val


# def match(m, s):
#     if isinstance(m, str):
#         m = ast.parse(m).body[0].value
#     if isinstance(s, str):
#         s = ast.parse(s).body[0].value
#     comp = Compare()
#     val = comp.compare(m, s)
#     d = comp.d
#     if len(d) == 0:
#         return val
#     else:
#         return d


# known_symbol_types = {
#     'dtcc': recipes['dtcc']['symbols'],
# }


# class ModelException(Exception):
#     type = 'error'


# def check_symbol_validity(s):
#     import ast
#     val = ast.parse(s).body[0].value
#     assert (isinstance(val, ast.Name))


# def check_symbols(data):

#     # can raise three types of exceptions
#     # - unknown symbol
#     # - invalid symbol
#     # - already declared

#     # add: not declared if missing 'states', 'controls' ?

#     exceptions = []

#     symbols = data['symbols']
#     cm_symbols = symbols
#     model_type = 'dtcc'

#     already_declared = {}  # symbol: symbol_type, position

#     for key, values in cm_symbols.items():
#         # (start_line, start_column, end_line, end_column) of the key
#         if key not in known_symbol_types[model_type]:
#             l0, c0, l1, c1 = cm_symbols.lc.data[key]
#             exc = ModelException(
#                 "Unknown symbol type '{}'".format(
#                     key, model_type))
#             exc.pos = (l0, c0, l1, c1)
#             # print(l0,c0,l1,c1)
#             exceptions.append(exc)
#             assert (isinstance(values, CommentedSeq))

#         for i, v in enumerate(values):
#             (l0, c0) = values.lc.data[i]
#             length = len(v)
#             l1 = l0
#             c1 = c0 + length
#             try:
#                 check_symbol_validity(v)
#             except:
#                 exc = ModelException("Invalid symbol '{}'".format(v))
#                 exc.pos = (l0, c0, l1, c1)
#                 exceptions.append(exc)
#             if v in already_declared:
#                 ll = already_declared[v]
#                 exc = ModelException(
#                     "Symbol '{}' already declared as '{}'. (pos {})".format(
#                         v, ll[0], (ll[1][0] + 1, ll[1][1])))
#                 exc.pos = (l0, c0, l1, c1)
#                 exceptions.append(exc)
#             else:
#                 already_declared[v] = (key, (l0, c0))

#     return exceptions


# def check_equations(data):

#     model_type = data['model_type']
#     pos0 = data.lc.data['equations']
#     equations = data['equations']

#     exceptions = []
#     recipe = recipes[model_type]
#     specs = recipe['specs']

#     for eq_type in specs.keys():
#         if (eq_type not in equations) and (not specs[eq_type].get(
#                 'optional', True)):
#             exc = ModelException("Missing equation type {}.".format(eq_type))
#             exc.pos = pos0
#             exceptions.append(exc)

#     already_declared = {}
#     unknown = []

#     for eq_type in equations.keys():
#         pos = equations.lc.data[eq_type]
#         if eq_type not in specs:
#             exc = ModelException("Unknown equation type {}.".format(eq_type))
#             exc.pos = pos
#             exceptions.append(exc)
#             unknown.append(eq_type)

#         # BUG: doesn't produce an error when a block is declared twice
#         # should be raised by ruaml.yaml ?
#         elif eq_type in already_declared.keys():
#             exc = ModelException(
#                 "Equation type {} declared twice at ({})".format(eq_type, pos))
#             exc.pos = pos
#             exceptions.append(exc)
#         else:
#             already_declared[eq_type] = pos

#     for eq_type in [k for k in equations.keys() if k not in unknown]:

#         for n, eq in enumerate(equations[eq_type]):
#             eq = eq.replace('<=', '<').replace('==',
#                                                '=').replace('=', '==').replace(
#                                                    '<', '<=')
#             # print(eq)
#             pos = equations[eq_type].lc.data[n]
#             try:
#                 ast.parse(eq)

#             except SyntaxError as e:
#                 exc = ModelException("Syntax Error.")
#                 exc.pos = [
#                     pos[0], pos[1] + e.offset, pos[0], pos[1] + e.offset
#                 ]
#                 exceptions.append(exc)

#         # TEMP: incorrect ordering
#         if specs[eq_type].get('target'):
#             for n, eq in enumerate(equations[eq_type]):
#                 eq = eq.replace('<=', '<').replace('==', '=').replace(
#                     '=', '==').replace('<', '<=')
#                 pos = equations[eq_type].lc.data[n]
#                 lhs_name = str.split(eq, '=')[0].strip()
#                 target = specs[eq_type]['target'][0]
#                 if lhs_name not in data['symbols'][target]:
#                     exc = ModelException(
#                         "Undeclared assignement target '{}'. Add it to '{}'.".
#                         format(lhs_name, target))
#                     exc.pos = [pos[0], pos[1], pos[0], pos[1] + len(lhs_name)]
#                     exceptions.append(exc)
#                 # if n>len(data['symbols'][target]):
#                 else:
#                     right_name = data['symbols'][target][n]
#                     if lhs_name != right_name:
#                         exc = ModelException(
#                             "Left hand side should be '{}' instead of '{}'.".
#                             format(right_name, lhs_name))
#                         exc.pos = [
#                             pos[0], pos[1], pos[0], pos[1] + len(lhs_name)
#                         ]
#                         exceptions.append(exc)
#         # temp
#     return exceptions


# def check_definitions(data):

#     if 'definitions' not in data:
#         return []
#     definitions = data['definitions']
#     if definitions is None:
#         return []

#     exceptions = []
#     known_symbols = sum([[*v] for v in data['symbols'].values()], [])

#     allowed_symbols = {v: (0, ) for v in known_symbols}  # TEMP
#     for p in data['symbols']['parameters']:
#         allowed_symbols[p] = (0, )

#     new_definitions = dict()
#     for k, v in definitions.items():
#         pos = definitions.lc.data[k]
#         if k in known_symbols:
#             exc = ModelException(
#                 'Symbol {} has already been defined as a model symbol.'.format(
#                     k))
#             exc.pos = pos
#             exceptions.append(exc)
#             continue
#         if k in new_definitions:
#             exc = ModelException(
#                 'Symbol {} cannot be defined twice.'.format(k))
#             exc.pos = pos
#             exceptions.append(exc)
#             continue
#         try:
#             check_symbol_validity(k)
#         except:
#             exc = ModelException("Invalid symbol '{}'".format(k))
#             exc.pos = pos
#             exceptions.append(exc)

#             # pos = equations[eq_type].lc.data[n]
#         try:
#             expr = ast.parse(str(v))

#             # print(allowed_symbols)
#             check = check_expression(expr, allowed_symbols)
#             # print(check['problems'])
#             for pb in check['problems']:
#                 name, t, offset, err_type = [pb[0], pb[1], pb[2], pb[3]]
#                 if err_type == 'timing_error':
#                     exc = Exception(
#                         'Timing for variable {} could not be determined.'.
#                         format(pb[0]))
#                 elif err_type == 'incorrect_timing':
#                     exc = Exception(
#                         'Variable {} cannot have time {}. (Allowed: {})'.
#                         format(name, t, pb[4]))
#                 elif err_type == 'unknown_function':
#                     exc = Exception(
#                         'Unknown variable/function {}.'.format(name))
#                 elif err_type == 'unknown_variable':
#                     exc = Exception(
#                         'Unknown variable/parameter {}.'.format(name))
#                 else:
#                     print(err_type)
#                 exc.pos = (pos[0], pos[1] + offset, pos[0],
#                            pos[1] + offset + len(name))
#                 exc.type = 'error'
#                 exceptions.append(exc)

#             new_definitions[k] = v

#             allowed_symbols[k] = (0, )  # TEMP
#             # allowed_symbols[k] = None

#         except SyntaxError as e:
#             pp = pos  # TODO: find right mark for pp
#             exc = ModelException("Syntax Error.")
#             exc.pos = [pp[0], pp[1] + e.offset, pp[0], pp[1] + e.offset]
#             exceptions.append(exc)

#     return exceptions


# def check_calibration(data):
#     # what happens here if symbols are not clean ?
#     symbols = data['symbols']
#     pos0 = data.lc.data['calibration']
#     calibration = data['calibration']
#     exceptions = []
#     all_symbols = []
#     for v in symbols.values():
#         all_symbols += v
#     for s in all_symbols:
#         if (s not in calibration.keys()) and (s not in symbols["exogenous"]):
#             # should skip invalid symbols there
#             exc = ModelException(
#                 "Symbol {} has no calibrated value.".format(s))
#             exc.pos = pos0
#             exc.type = 'warning'
#             exceptions.append(exc)
#     for s in calibration.keys():
#         val = str(calibration[s])
#         try:
#             ast.parse(val)
#         except SyntaxError as e:
#             pos = calibration.lc.data[s]
#             exc = ModelException("Syntax Error.")
#             exc.pos = [pos[0], pos[1] + e.offset, pos[0], pos[1] + e.offset]
#             exceptions.append(exc)
#     return exceptions


# def check_all(data):
#     def serious(exsc):
#         return ('error' in [e.type for e in exsc])

#     exceptions = check_infos(data)
#     if serious(exceptions):
#         return exceptions
#     exceptions = check_symbols(data)
#     if serious(exceptions):
#         return exceptions
#     exceptions += check_definitions(data)
#     if serious(exceptions):
#         return exceptions
#     exceptions += check_equations(data)
#     if serious(exceptions):
#         return exceptions
#     exceptions += check_calibration(data)
#     if serious(exceptions):
#         return exceptions
#     return exceptions


# def human_format(err):
#     err_type = err['type']
#     err_type = colored(
#         err_type, color=('red' if err_type == 'error' else 'yellow'))
#     err_range = str([e + 1 for e in err['range'][0]])[1:-1]
#     return '{:7}: {:6}: {}'.format(err_type, err_range, err['text'])


# def check_infos(data):
#     exceptions = []
#     if 'model_type' in data:
#         model_type = data['model_type']
#         if model_type not in ['dtcc', 'dtmscc', 'dtcscc', 'dynare']:
#             exc = ModelException('Uknown model type: {}.'.format(
#                 str(model_type)))
#             exc.pos = data.lc.data['model_type']
#             exc.type = 'error'
#             exceptions.append(exc)
#     else:
#         model_type = 'dtcc'
#         data['model_type'] = 'dtcc'
#         # exc = ModelException("Missing field: 'model_type'.")
#         # exc.pos = (0,0,0,0)
#         # exc.type='error'
#         # exceptions.append(exc)
#     if 'name' not in data:
#         exc = ModelException("Missing field: 'name'.")
#         exc.pos = (0, 0, 0, 0)
#         exc.type = 'warning'
#         exceptions.append(exc)
#     return exceptions


# def lint(txt, source='<string>', format='human', catch_exception=False):

#     # raise ModelException if it doesn't work correctly
#     if isinstance(txt, str):
#         try:
#             data = ry.load(txt, ry.RoundTripLoader)
#         except Exception as exc:
#             if not catch_exception:
#                 raise exc
#             return []  # should return parse error
#     else:
#         # txt is then assumed to be a ruamel structure
#         data = txt

#     if not ('symbols' in data or 'equations' in data or 'calibration' in data):
#         # this is probably not a yaml filename
#         output = []
#     else:
#         try:
#             exceptions = check_all(data)
#         except Exception as e:
#             if not catch_exception:
#                 raise(e)
#             exc = ModelException("Linter Error: Uncaught Exception.")
#             exc.pos = [0, 0, 0, 0]
#             exc.type = 'error'
#             exceptions = [exc]

#         output = []
#         for k in exceptions:
#             try:
#                 err_type = k.type
#             except:
#                 err_type = 'error'
#             output.append({
#                 'type':
#                 err_type,
#                 'source':
#                 source,
#                 'range': ((k.pos[0], k.pos[1]), (k.pos[2], k.pos[3])),
#                 'text':
#                 k.args[0]
#             })

#     if format == 'json':
#         return (json.dumps(output))
#     elif format == 'human':
#         return (str.join("\n", [human_format(e) for e in output]))
#     elif not format:
#         return output
#     else:
#         raise ModelException("Unkown format {}.".format(format))


# TODO:
# - check name (already defined by smbdy else ?)
# - description: ?
# - calibration:
#      - incorrect key
#          - warning if not a known symbol ?
#          - not a recognized identifier
#          - defined twice
#      - impossible to solve in closed form (depends on ...)
#      - incorrect equation
#           - grammatically incorrect
#           - contains timed variables
#      - warnings:
#           - missing values
# - equations: symbols already known (beware of speed issues)
#     - unknown group of equations
#     - incorrect syntax
#     - undeclared variable (and not a function)
#     - indexed parameter
#     - incorrect order
#     - incorrect complementarities
#     - incorrect recipe: unexpected symbol type
#     - nonzero residuals (warning, to be done without compiling)
# - options: if present
#     - approximation_space:
#          - inconsistent boundaries
#                - must equal number of states
#     - distribution:
#          - same size as shocks
