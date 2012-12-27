from setuptools import setup,find_packages
import py2exe

from dolo import __version__

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
    version = __version__,

    # Declare your packages' dependencies here, for eg:

    # Fill in these to make your Egg ready for upload to
    # PyPI
    author = 'Pablo Winant',
    author_email = '',

    url = 'www.mosphere.fr',
    license = 'BSD-2',
	
	packages = ['dolo'],

    scripts = ['bin/dolo-matlab','bin/dolo-recs'],
 
	data_files = data_files,
 
    options = {
            "py2exe" : {
                #"includes" : ["sip"],
				"excludes" : excludes,
                "dist_dir" : "../windist/",
                "bundle_files" : 3
            }
    },
    console = ['bin/dolo-matlab.py'],

)
