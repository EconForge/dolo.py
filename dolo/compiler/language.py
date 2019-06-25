import numpy
import math

functions = {
    'log': math.log,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.cos,
    'tan': math.tan,
}

constants = {
    'pi': math.pi
}

# objects

from dolo.compiler import objects

class LanguageElement(dict):

    @classmethod
    def constructor(cls, loader, node):
        value = loader.construct_mapping(node)
        print("Constructing")
        print(value.__class__)
        return cls(**value)

    def check(self):
        import inspect
        sig = inspect.signature(self.baseclass)
        sig.bind_partial(self)

    def eval(self, d={}):
        from dolo.compiler.symbolic_eval import NumericEval
        ne = NumericEval(d)
        args = ne.eval_dict(self)
        obj = self.baseclass(**args)
        return obj

    def __repr__(self):
        s = super().__repr__()
        n = self.baseclass.__name__
        return "{}(**{})".format(n, s)

    def __str__(self):
        n = self.baseclass.__name__
        c = str.join(", ", ["{}={}".format(k, v) for k, v in self.items()])
        return "{}({})".format(n, c)


class Normal(LanguageElement):
    baseclass = objects.Normal

class AR1(LanguageElement):
    baseclass = objects.AR1

class MarkovChain(LanguageElement):
    baseclass = objects.MarkovChain
#
class MarkovProduct(LanguageElement):
    baseclass = objects.MarkovProduct

class CartesianGrid(LanguageElement):
    baseclass = objects.CartesianGrid

class SmolyakGrid(LanguageElement):
    baseclass = objects.SmolyakGrid

class Domain(LanguageElement):
    baseclass = objects.Domain

from dolo.numeric.processes import ConstantProcess, AggregateProcess, Product
from dolo.numeric.processes_iid import UNormal, Uniform

class UNormal(LanguageElement):
    baseclass = UNormal

class Uniform(LanguageElement):
    baseclass = Uniform

class ConstantProcess(LanguageElement):
    baseclass = ConstantProcess

class AggregateProcess(LanguageElement):
    baseclass = AggregateProcess

class Product(LanguageElement):
    baseclass = Product

# aliases
class Smolyak(SmolyakGrid):
    pass

class Cartesian(CartesianGrid):
    pass

minilang = [
    Normal,
    AR1,
    MarkovChain,
    MarkovProduct,
    Smolyak,
    Cartesian,
    CartesianGrid,
    SmolyakGrid,
    Domain,
    UNormal,
    Uniform,
    ConstantProcess,
    AggregateProcess,
    Product
]


if __name__ == '__main__':


    txt = """
    grid: !Cartesian
       a: [x,10]
       b: [y,20]
       orders: [50,50]

    distribution: !Normal
        Sigma: [[sig_z]]
        mu: [0.0]
    """


    data = yaml.load(txt)


    dis = data['distribution']
    str(dis)


    dis.__repr__()
    grid = data['grid']


    d = dict(x=20, y=30, sig_z=0.001)

    from dolo.compiler.symbolic_eval import NumericEval

    ne = NumericEval(d, minilang=minilang)

    ne.eval(d)

    cart = grid.eval(d)
    dd = dis.eval(d)

    ne.eval(data['grid'])

    ndata = ne.eval(data)

    data['grid']
    ndata['grid']
