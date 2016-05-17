import numpy
import math

functions = {
    'log': math.log,
    'exp': math.exp,
    'sin': math.sin,
    'cos': math.sin,
    'tan': math.sin,
}

constants = {
    'pi': math.pi
}

# objects



class LanguageElement(dict):

    @classmethod
    def constructor(cls, loader, node):
        value = loader.construct_mapping(node)
        return cls(**value)


class Normal(LanguageElement):
    pass

class Cartesian(LanguageElement):
    pass

class AR1(LanguageElement):
    pass

minilang = {
    'Normal': Normal,
    'Cartesian': Cartesian,
    'AR1': AR1
}

# if __name__ == '__main__':

# import ruamel.yaml as yaml
import yaml

for k,C in minilang.items():
    yaml.add_constructor('!{}'.format(k), C.constructor)
# yaml.add_constructor('!Cartesian', constructor_cartesian)


txt = """
grid: !Cartesian
   a: [x,10]
   b: [y,20]
   orders: [50,50]

distribution: !Normal
    sigma: [[sig_z]]
    mu: [0.0]
"""
data = yaml.load(txt)


dis = data['distribution']
dis


grid = data['grid']

grid

from dolo.compiler.symbolic_eval import NumericEval

d = dict(x=20, y=30)

ne = NumericEval(d)
ne.eval(data)
