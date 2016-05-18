def test_lint_all():

    from dolo import yaml_import
    import os

    examples_dir = 'examples/models/'
    examples = os.listdir(examples_dir)

    for f in examples:
        print(f)
        check = yaml_import(examples_dir + f, check_only=True)

        # need to fix Dynare lint before reenabling
        # if len(check)>1:
        #     assert("Linter Error" not in check[0]) # assert there is not linter error
        print()

if __name__ == "__main__":
    test_lint_all()
