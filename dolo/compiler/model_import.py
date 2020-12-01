import numpy

from dolo.misc.display import read_file_or_url
import yaml


def yaml_import(fname, check=True, check_only=False):

    txt = read_file_or_url(fname)

    try:
        data = yaml.compose(txt)
        # print(data)
        # return data
    except Exception as ex:
        print(
            "Error while parsing YAML file. Probable YAML syntax error in file : ",
            fname,
        )
        raise ex

    # if check:
    #     from dolo.linter import lint
    #     data = ry.load(txt, ry.RoundTripLoader)
    #     output = lint(data, source=fname)
    #     if len(output) > 0:
    #         print(output)

    # if check_only:
    #     return output

    data["filename"] = fname

    from dolo.compiler.model import Model

    return Model(data, check=check)
    # from dolo.compiler.model import SymbolicModel
