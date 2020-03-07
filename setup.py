from setuptools import setup, find_packages

# get version number
exec(open('dolo/version.py').read())

setup(
    name="dolo",
    version=__version__,
    packages=find_packages(),
    package_data={'dolo.compiler': ["recipes.yaml"]},
    test_suite='dolo.tests',
    scripts=[
        'bin/dolo-recs', 'bin/dolo-matlab', 'bin/dolo-julia', 'bin/dolo',
        'bin/dolo-lint'
    ],
    install_requires=[
        "dolang", "pyyaml", "numba", "numpy", "sympy", "scipy", "quantecon", "pandas",
        "interpolation", "ruamel.yaml", "xarray", "altair", "multipledispatch", "multimethod"
    ],
    extras_require={
        'interactive': ['ipython'],
        'plots': ["matplotlib"],
    },
    author="Pablo Winant",
    author_email="pablo.winant@gmail.com",
    description='Economic modelling in Python',
    license='BSD-2',
    url='http://albop.github.com/dolo/',
)
