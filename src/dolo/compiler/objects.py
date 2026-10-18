from dolo.numeric.distribution import *

#
from dataclasses import dataclass
from dolang.language import language_element

# not sure we'll keep that
import numpy as np
from typing import List, Union

Scalar = Union[int, float]

# not really a language element though
# @language_element


from dolo.numeric.spaces import Space, CartesianSpace

Domain = Space


class CartesianDomain(CartesianSpace):
    """Backwards-compatible alias for CartesianSpace."""

    pass


# these are dummy objects so far
#
# @language_element
# @dataclass
# class UNormal:
#     mu: float
#     sigma: float
#     signature = {'mu': 'float', 'sigma': 'float'}
#


# @language_element
# @dataclass
# class MvNormal:
#     Mu: List[float]
#     Sigma: List[List[float]]
#     signature = {'Mu': 'list(float)', 'Sigma': 'Matrix'}


# %%


@language_element
class Conditional:

    signature = {"condition": None, "type": None, "arguments": None}

    def __init__(self, condition, type, arguments):
        self.condition = condition
        self.type = type
        self.arguments = arguments


@language_element
class Product:
    def __init__(self, *args: List):
        self.factors = args


@language_element
def Matrix(*lines):
    mat = np.array(lines, np.float64)
    assert mat.ndim == 2
    return mat


@language_element
def Vector(*elements):
    mat = np.array(elements, np.float64)
    assert mat.ndim == 1
    return mat
