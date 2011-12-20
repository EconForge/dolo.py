from setuptools import setup, find_packages

__version__ = "0.3-dev"

setup(
    name = "dolo",
    version = __version__,
    packages = find_packages('src'),
    package_dir = {'':'src'},
    scripts = ['src/bin/dolo-recs.py', 'src/bin/dolo-matlab.py'],

    author = "Pablo Winant",
    author_email = "pablo.winant@gmail.com",
    description = 'DSGE modelling library',
    license = 'BSD',
    url = 'http://code.google.com/p/dynare-python/',

)

