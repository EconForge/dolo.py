
from dolo.compiler.import_dynare import import_dynare
import os


def test_dynare_import():

    examples_dir = "examples/dynare_modfiles/"
    listdir = os.listdir(examples_dir)

    failed = []
    for fname in listdir:
        try:
            model = import_dynare(examples_dir + fname)
        except Exception as e:
            failed.append((fname, e))

    print("Failed:")
    for f in failed: print(f)


if __name__ == '__main__':
    test_dynare_import()
