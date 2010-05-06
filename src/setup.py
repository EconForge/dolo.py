__author__="pablo"
__date__ ="$3 sept. 2009 11:26:05$"

from setuptools import setup,find_packages
import py2xe

setup (
  name = 'dolo',
  version = '0.1',
  packages = find_packages(),

  # Declare your packages' dependencies here, for eg:
  install_requires=[],

  # Fill in these to make your Egg ready for upload to
  # PyPI
  author = 'pablo',
  author_email = '',

  summary = 'Just another Python package for the cheese shop',
  url = '',
  license = '',
  long_description= 'Long description of the package',

  # could also include long_description, download_url, classifiers, etc.

  console = 'doloc.py'
)
