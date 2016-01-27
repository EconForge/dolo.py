

#from __future__ import print_function

# This module is supposed to be imported first
# it contains global variables used for configuration


# try to register printing methods if IPython is running

save_plots = False
real_type = 'double'

import warnings

from dolo.misc.termcolor import colored


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '{}:{}:{}\n    {}\n'.format(
            colored( category.__name__, 'yellow'),
            filename,
            lineno,
            message)
warnings.formatwarning = warning_on_one_line
import numpy
numpy.seterr(all='ignore')


# create temporary directory for compiled functions
import tempfile, sys
temp_dir = tempfile.mkdtemp(prefix='dolo_')
sys.path.append(temp_dir)


from IPython.core.display import display
