#!/usr/bin/env python


from distutils.core import setup
from distutils.core import Command
import sys

from dolo import __version__

# Make sure I have the right Python version.
if sys.version_info[1] < 6:
    print "Dolo requires Python 2.6 or newer. Python %d.%d detected" % \
          sys.version_info[:2]
    sys.exit(-1)


class clean(Command):
    """Cleans *.pyc and debian trashs, so you should get the same copy as
    is in the svn.
    """
    user_options = []

    description = "Clean everything"
#    user_options = [("all","a","the same")]

    def initialize_options(self):
        self.all = None

    def finalize_options(self):
        pass

    def run(self):
        import os
        os.system("rm -rf build")
        os.system("rm -rf dist")

setup(
      name = 'dolo',
      version = __version__,
      description = 'DSGE modelling library',
      license = 'GPL3',
      url = 'http://code.google.com/p/dynare-python/',
      packages = ['dolo','dolo.symbolic','dolo.misc','dolo.numeric','dolo.numeric.extern','dolo.compiler'],
      scripts = [],
      ext_modules = [],
      data_files = [],
      cmdclass    = {
                     'clean' : clean,
                     },
      )

