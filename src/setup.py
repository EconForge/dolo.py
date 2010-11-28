
__author__="pablo"
__date__ ="$3 sept. 2009 11:26:05$"

from setuptools import setup,find_packages
import py2exe

print find_packages()

setup (
    name = 'dolo',
    version = '0.2',
    packages = find_packages(),

    # Declare your packages' dependencies here, for eg:

    # Fill in these to make your Egg ready for upload to
    # PyPI
    author = 'pablo',
    author_email = '',

    summary = 'Just another Python package for the cheese shop',
    url = '',
    license = '',
    long_description= 'Long description of the package',
    options = {
            "py2exe" : {
                "includes" : ["sip"],
                
            }
    },
    console = ['bin/doloc.py'],
    windows = ['bin/dyngui.py'],

    )
