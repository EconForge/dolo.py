from dolo import yaml_import
from dolo import dprint
#
# def test_ar1():
#     print("Test AR1 import")
#     model = yaml_import('examples/models/rbc_dtcc_ar1.yaml')
#
# def test_iid():
#     print('Test IID import')
#     model = yaml_import('examples/models/rbc_dtcc_iid.yaml')
#     dprint(model.exogenous)
#     dprint(model.distribution)
#
# def test_mc():
#     print('Test MC import')
#     model = yaml_import('examples/models/rbc_dtcc_mc.yaml')
#     dprint(model.exogenous)
#     dprint(model.discrete_transition)

def test_dtcscc():
    print('Test DTCSCC import')
    model = yaml_import('examples/models/compat/rbc.yaml')
    dprint(model.exogenous)
    dprint(model.discrete_transition)
    dprint(model.is_dtcscc())
