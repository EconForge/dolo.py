from setuptools import setup, find_packages, Extension

__version__ = "0.4-dev-1"

setup(

    name = "dolo",
    version = __version__,
    packages = ['dolo'],

    test_suite='tests',
    
    scripts = ['bin/dolo-recs.py', 'bin/dolo-matlab.py'],

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

