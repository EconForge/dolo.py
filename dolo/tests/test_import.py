from dolo import *

from dolo.algos.value_iteration import constant_policy

def test_import():

    filenames = [
        "examples/models/rbc_iid.yaml",
        "examples/models/rbc_mc.yaml",
        # "examples/models/rbc_ar1.yaml"
    ]

    for fname in filenames:

        model = yaml_import(fname, check=False)
        print(model)
        print("Exogenous shocks:")
        print(model.exogenous)
        print("Discretized shock:")
        print(model.exogenous.discretize())
        try:
            print("Distribution;")
            print(model.get_distribution())
        except:
            pass

        dprocess = model.exogenous.discretize()

        print( dprocess.n_nodes )
        print( dprocess.n_inodes(0) )
        print( dprocess.inode(0,0) )
        print( dprocess.node(0) )

if __name__ == "__main__":
    test_import()
