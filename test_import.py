import ruamel.yaml as ry
from ruamel.yaml.comments import CommentedSeq, CommentedMap
#
# class OrderedLoader(Loader):
#     pass
# def construct_mapping(loader, node):
#     loader.flatten_mapping(node)
#     return object_pairs_hook(loader.construct_pairs(node))
# OrderedLoader.add_constructor(
#     yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
#     construct_mapping)
# return yaml.load(stream, OrderedLo


with open("examples/models/rbc00.yaml") as f:
    txt = f.read()


print( txt )

txt = txt.replace("^","**")

data = ry.load(txt, ry.RoundTripLoader)

data.__class__

data

class SymbolicModel:

    def __init__(self, data):

        self.data = data

    @property
    def symbols(self):
        return self.data['symbols']

    @property
    def equations(self):
        return self.data['equations']

    @property
    def infos(self):
        return self.data['infos']

    def get_calibration(self):

        calibration = self.data.get("calibration", {})
        from dolo.compiler.triangular_solver import solve_triangular_system
        return solve_triangular_system(calibration)

    def get_domain(self):

        sdomain = self.data.get("domain", {})
        calibration = self.get_calibration()
        states = self.symbols['states']

        from dolo.compiler.language import Domain
        d = Domain(**sdomain)
        domain = d.eval(d=calibration)
        # a bit of a hack...
        for k in domain.keys():
            if k not in states:
                domain.pop(k)
        return domain

    def get_exogenous(self):

        exo = self.data.get("exogenous", {})
        calibration = self.get_calibration()
        type = get_type(exo)
        from dolo.compiler.language import Normal, AR1
        if type=="Normal":
            exog = Normal(**exo)
        elif type=="AR1":
            exog = AR1(**exo)
        elif type=="MarkovChain":
            exog = MarkovChain(**exo)
        d = exog.eval(d=calibration)
        return d


    def get_grid(self):

        options = self.data.get("options", {})
        options.get("grid",{})


def get_type(d):
    try:
        s = d.tag.value
        return s.strip("!")
    except:
        v = d.get("type")
        return v

s = "!ji"

s.strip("!")

smodel = SymbolicModel(data)
smodel.data.get('symbols')
smodel.get_calibration()
smodel.get_domain()
smodel.get_exogenous()
