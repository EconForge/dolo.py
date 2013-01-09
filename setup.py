from setuptools import setup, find_packages
from distutils.extension import Extension

#from Cython.Distutils import build_ext

#from dolo import __version__

__version__ = '0.4.4'

setup(

    name = "dolo",
    version = __version__,
    packages = find_packages(),

    test_suite='dolo.tests',

#    cmdclass = {'build_ext': build_ext},
#    ext_modules = [
#        Extension('dolo.numeric.serial_operations_cython',['dolo/numeric/serial_operations_cython.pyx']),
#        Extension('dolo.numeric.interpolation.splines_cython',['dolo/numeric/interpolation/splines_cython.pyx'], library_dirs = ['/usr/local/lib'], include_dirs = ['/usr/local/include/einspline'], libraries = ['m','einspline'])
#    ],
    
    scripts = ['bin/dolo-recs', 'bin/dolo-matlab', 'bin/dolo-julia', 'bin/dolo'],

    install_requires = ["pyyaml","sympy","numpy"],

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

