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

        calibration = self.get_calibration()

        sdomain = self.data.get("domain", {})
        for k in sdomain.keys():
            if k not in states:
                sdomain.pop(k)

        states = self.symbols['states']

        from dolo.compiler.language import Domain
        d = Domain(**sdomain)
        domain = d.eval(d=calibration)
        d.states = states
        # a bit of a hack...

        return domain

    def get_exogenous(self):

        exo = self.data.get("exogenous", {})
        calibration = self.get_calibration()
        type = get_type(exo)
        from dolo.compiler.language import Normal, AR1, MarkovChain
        if type == "Normal":
            exog = Normal(**exo)
        elif type == "AR1":
            exog = AR1(**exo)
        elif type == "MarkovChain":
            exog = MarkovChain(**exo)
        d = exog.eval(d=calibration)
        return d


    def get_grid(self):

        options = self.data.get("options", {})

        # determine grid_type
        grid_type = get_type(options.get("grid"))
        if grid_type is None:
            grid_type = get_address(self.data, "options:grid:type", "options:grid_type")
        if grid_type is None:
            raise Exception('Missing grid geometry ("options:grid:type")')

        # determine bounds:
        domain = self.get_domain()
        min = domain.min
        max = domain.max

        print(min, max)
        # if grid_type:
        #     grid["type"] = grid_type
        # if len(grid) is None:
        #     # create default grid
        #
        # print(grid_type)


def get_type(d):
    try:
        s = d.tag.value
        return s.strip("!")
    except:
        v = d.get("type")
        return v

def get_address(data, address):
    if isinstance(address, list):
        if list(address)==0:
            return None
        resp = get_address(data, address[0])
        if resp is not None:
            return resp
        else:
            return get_address(data, addresss[1:])
    fields = str.split(address, ':')
    while len(fields)>0:
        data = data.get(fields[0])
        fields = fields[1:]
        if data is None:
            return data
    return data


smodel = SymbolicModel(data)
smodel.data.get('symbols')
smodel.get_calibration()
smodel.get_domain()
smodel.get_exogenous()

print( smodel.get_grid() )
