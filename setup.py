from setuptools import setup, find_packages
from distutils.extension import Extension

import numpy as np

# get version number
exec( open('dolo/version.py').read())

setup(

    name = "dolo",
    version = __version__,
    packages = find_packages(),

    package_data={'dolo.compiler':["recipes.yaml"]},

    test_suite='dolo.tests',

    include_dirs = [np.get_include()],

    scripts = ['bin/dolo-recs', 'bin/dolo-matlab', 'bin/dolo-julia', 'bin/dolo', 'bin/dolo-lint'],

    install_requires = ["pyyaml", "numba>=0.13", "numpy", "numexpr", "sympy",
                        "pandas", "interpolation"],

    extras_require = {
            'plots':  ["matplotlib"],
            'higher_order': ["slycot"]
    },

    author = "Pablo Winant",
    author_email = "pablo.winant@gmail.com",

    description = 'Economic modelling in Python',

    license = 'BSD-2',

    url = 'http://albop.github.com/dolo/',

)
