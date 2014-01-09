

#from __future__ import print_function

# This module is supposed to be imported first
# it contains global variables used for configuration


# try to register printing methods if IPython is running

save_plots = False
real_type = 'double'

from dolo.misc.termcolor import colored
import warnings

def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '{}:{}:{}\n    {}\n'.format(
            colored( category.__name__, 'yellow'),
            filename,
            lineno,
            message)
warnings.formatwarning = warning_on_one_line

try:
    import dolo.misc.printing as printing
    from numpy import ndarray
    from dolo.symbolic.model import SModel
    from dolo.numeric.decision_rules import DynareDecisionRule
    from dolo.compiler.compiler_python import GModel, GModel_fg_from_fga

    ip = get_ipython()

    # there could be some kind of auto-discovery there
    ip.display_formatter.formatters['text/html'].for_type( GModel, printing.print_cmodel )
    ip.display_formatter.formatters['text/html'].for_type( GModel_fg_from_fga, printing.print_cmodel )
    ip.display_formatter.formatters['text/html'].for_type( ndarray, printing.print_array )
    ip.display_formatter.formatters['text/html'].for_type( SModel, printing.print_model )
    ip.display_formatter.formatters['text/html'].for_type( DynareDecisionRule, printing.print_dynare_decision_rule )


    from IPython.core.display import display

except:

    from pprint import pprint
    def display(txt):
        pprint(txt)
