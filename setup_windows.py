
__author__="pablo"
__date__ ="$3 sept. 2009 11:26:05$"

from setuptools import setup,find_packages
import py2exe

from dolo import __version__ as __dolo_version__
from glob import glob

data_files = [(
	"Microsoft.VC90.CRT",
	glob("C:\\Program Files\\pythonxy\\console\\Microsoft.VC90.CRT\\*.*")
)]

excludes = ["pywin", "pywin.debugger", "pywin.debugger.dbgcon",
            "pywin.dialogs", "pywin.dialogs.list",
            "Tkconstants","Tkinter","tcl"]

import sys
sys.path.append("C:\\Program Files\\pythonxy\\console\\Microsoft.VC90.CRT\\")
setup (
    name = 'dolo',
    #version = '0.3',

    # Declare your packages' dependencies here, for eg:

    # Fill in these to make your Egg ready for upload to
    # PyPI
    author = 'Pablo Winant',
    author_email = '',

    url = 'www.mosphere.fr',
    license = 'BSD',
	
	packages = find_packages('src'),
    package_dir = {'':'src'},
    scripts = ['src/bin/dolo-matlab.py'],
 
	data_files = data_files,
 
    options = {
            "py2exe" : {
                #"includes" : ["sip"],
				"excludes" : excludes,
                "dist_dir" : "../windist/",
                "bundle_files" : 3
            }
    },
    console = ['src/bin/dolo-matlab.py'],
    )
