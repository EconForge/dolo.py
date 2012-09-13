from setuptools import setup, find_packages

__version__ = "0.4-dev"

setup(
    name = "dolo",
    version = __version__,
    packages = find_packages('src'),
    package_dir = {'':'src'},
    scripts = ['src/bin/dolo-recs.py', 'src/bin/dolo-matlab.py'],
    install_requires = ["pyyaml","sympy","numpy","matplotlib"],
    author = "Pablo Winant",
    author_email = "pablo.winant@gmail.com",
    description = 'Economic modelling in Python',
    license = 'BSD-2',
    url = 'http://albop.github.com/dolo/',

)

