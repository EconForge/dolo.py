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


def yaml_import_v2(fname, filename=None, check=True):
    """Import a dolo YAML model using the dynspec-based importer (ModelV2)."""
    from dolo.compiler.model_v2 import yaml_compose, ModelV2

    txt = read_file_or_url(fname)
    data = yaml_compose(txt)
    if filename is None:
        filename = str(fname)
    return ModelV2.from_yaml(data, filename=filename, check=check)


def dyno_import(fname, filename=None, check=True):
    """Import a ``.dyno`` model file using the dynspec-based importer (ModelV2)."""
    from dolo.compiler.model_v2 import ModelV2

    txt = read_file_or_url(fname)
    if filename is None:
        filename = str(fname)
    return ModelV2.from_dyno(txt, filename=filename, check=check)
