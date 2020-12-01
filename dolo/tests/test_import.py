from dolo import *

from dolo.algos.value_iteration import constant_policy


def test_import():

    filenames = [
        "examples/models/rbc_iid.yaml",
        "examples/models/rbc_mc.yaml",
        "examples/models/rbc_ar1.yaml",
        "examples/models/rbc.yaml",
    ]

    for fname in filenames:

        model = yaml_import(fname, check=False)
        print(model)
        print("Exogenous shocks:")
        print(model.exogenous)
        print("Discretized shock:")
        print(model.exogenous.discretize())
        print(model.symbols)
        print(model.definitions)
        dprocess = model.exogenous.discretize()
        print(dprocess.n_nodes)
        print(dprocess.n_inodes(0))
        print(dprocess.inode(0, 0))
        print(dprocess.node(0))


def test_old_models():

    import os

    os.listdir("examples/models_")  # old models
    filenames = [
        f"examples/models_/{fname}" for fname in os.listdir("examples/models_")
    ]

    for fname in filenames:

        try:
            print(f"Importing: {fname}")
            model = yaml_import(fname, check=True)

        except Exception as e:
            print(fname)
            raise (e)
            # assert( not isinstance(e, Exception) )


if __name__ == "__main__":
    test_import()
