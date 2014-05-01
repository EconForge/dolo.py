

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

# try:
#     from numpy import ndarray
#     from dolo.compiler.model_numeric import NumericModel
#
#     ip = get_ipython()
#
#     # there could be some kind of auto-discovery there
#     # ip.display_formatter.formatters['text/html'].for_type( NumericModel, lambda m: m.__str__() )

from IPython.core.display import display
