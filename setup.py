from setuptools import setup, find_packages
from distutils.extension import Extension

from Cython.Distutils import build_ext
import numpy as np

from dolo import __version__

setup(

    name = "dolo",
    version = __version__,
    packages = find_packages(),

    test_suite='dolo.tests',

    cmdclass = {'build_ext': build_ext},

    ext_modules = [

        Extension(
		    'dolo.numeric.interpolation.multilinear_cython',
		    ['dolo/numeric/interpolation/multilinear_cython.pyx'],
            extra_compile_args=['-O3']
        ),

        Extension(
            'dolo.numeric.interpolation.serial_operations_cython',
            ['dolo/numeric/serial_operations_cython.pyx'],
            extra_compile_args=['-O3']
        ),

    ],

    include_dirs = [np.get_include()],
    
    scripts = ['bin/dolo-recs', 'bin/dolo-matlab', 'bin/dolo-julia', 'bin/dolo'],

    install_requires = ["pyyaml","sympy","numpy","cython"],

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

