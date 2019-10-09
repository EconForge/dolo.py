import numpy
import ruamel.yaml as ry

from dolo.misc.display import read_file_or_url
import yaml


def yaml_import(fname, check=True, check_only=False):

    txt = read_file_or_url(fname)
    txt = txt.replace('^', '**')

    try:
        data = ry.load(txt, ry.RoundTripLoader)
    except Exception as ex:
        print ("Error while parsing YAML file. Probable YAML syntax error in file : ", fname )
        raise ex

    if check:
        from dolo.linter import lint
        output = lint(data, source=fname)
        if len(output) > 0:
            print(output)

    if check_only:
        return output

    data['filename'] = fname

    from dolo.compiler.model import Model

    return Model(data)
    # from dolo.compiler.model import SymbolicModel


if __name__ == "__main__":

    # fname = "../../examples/models/compat/rbc.yaml"
    fname = "examples/models/compat/integration_A.yaml"

    import os
    print(os.getcwd())

    model = yaml_import(fname)

    print("calib")
    # print(model.calibration['parameters'])

    print(model)

    print(model.get_calibration(['beta']))
    model.set_calibration(beta=0.95)

    print(model.get_calibration(['beta']))

    print(model)

    s = model.calibration['states'][None, :]
    x = model.calibration['controls'][None, :]
    e = model.calibration['shocks'][None, :]

    p = model.calibration['parameters'][None, :]

    S = model.functions['transition'](s, x, e, p)
    lb = model.functions['controls_lb'](s, p)
    ub = model.functions['controls_ub'](s, p)

    print(S)

    print(lb)
    print(ub)
