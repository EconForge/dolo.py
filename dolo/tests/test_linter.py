# def test_lint_all():

#     from dolo import yaml_import
#     import os
#     import re

#     examples_dir = 'examples/models/'
#     examples = os.listdir(examples_dir)

#     regex = re.compile("(.*).yaml")

#     for f in examples:
#         if not regex.match(f):
#             continue
#         print("Checking: {}".format(f))
#         check = yaml_import(examples_dir + f, check_only=True)

#         # TODO: need to fix Dynare lint before reenabling
#         # if len(check)>1:
#         #     assert("Linter Error" not in check[0]) # assert there is no linter error
#         print()

# if __name__ == "__main__":
#     test_lint_all()
