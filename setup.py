from setuptools import setup, find_packages
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np

# get version number
exec( open('dolo/version.py').read())

setup(

    name = "dolo",
    version = __version__,
    packages = find_packages(),

    package_data={'dolo.symbolic':["recipes.yaml"]},

    test_suite='dolo.tests',

    include_dirs = [np.get_include()],
    
    scripts = ['bin/dolo-recs', 'bin/dolo-matlab', 'bin/dolo-julia', 'bin/dolo'],

    install_requires = ["pyyaml","sympy<=0.7.3","numpy","cython"],

    extras_require = {
            'plots':  ["matplotlib"],
            'first order solution':  ["scipy"],
            'higher order solution':  ["Slycot"],
    },

    author = "Pablo Winant",
    author_email = "pablo.winant@gmail.com",

    description = 'Economic modelling in Python',

    license = 'BSD-2',

    url = 'http://albop.github.com/dolo/',

)

