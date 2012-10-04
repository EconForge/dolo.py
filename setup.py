from setuptools import setup, find_packages

#from dolo import __version__

__version__ = '0.4-dev-2'

packages = find_packages('.', exclude='*tests*')
print(packages)

setup(

    name = "dolo",
    version = __version__,
    packages = packages, 

    test_suite='dolo.tests',
    
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

