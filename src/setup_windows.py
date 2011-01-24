
__author__="pablo"
__date__ ="$3 sept. 2009 11:26:05$"

from setuptools import setup,find_packages
import py2exe

from dolo import __version__ as __dolo_version__

setup (
    name = 'dolo',
    version = '0.2',
    packages = find_packages(),

    # Declare your packages' dependencies here, for eg:

    # Fill in these to make your Egg ready for upload to
    # PyPI
    author = 'pablo',
    author_email = '',

    url = '',
    license = '',
    long_description= 'Long description of the package',
    options = {
            "py2exe" : {
                "includes" : ["sip"],
                "dist_dir" : "../windist/",
                "bundle_files" : 1
            }
    },
    data_files = [('',['bin/options.ui','bin/modfile_editor.ui','bin/equation_widget.ui'])],
    console = ['bin/doloc.py'],
    windows = ['bin/dyngui.py']
    )
