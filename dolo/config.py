#from __future__ import print_function

# This module is supposed to be imported first
# it contains global variables used for configuration


# try to register printing methods if IPython is running

save_plots = False

try:
    import dolo.misc.printing as printing
    from numpy import ndarray
    from dolo.symbolic.model import Model
    from dolo.numeric.decision_rules import DynareDecisionRule

    ip = get_ipython()

    # there could be some kind of autodecovery there
    ip.display_formatter.formatters['text/html'].for_type( ndarray, printing.print_array )
    ip.display_formatter.formatters['text/html'].for_type( Model, printing.print_model )
    ip.display_formatter.formatters['text/html'].for_type( DynareDecisionRule, printing.print_dynare_decision_rule )


    from IPython.core.display import display

except:

    print("failing back on pretty_print")

    from pprint import pprint
    def display(txt):
        pprint(txt)
