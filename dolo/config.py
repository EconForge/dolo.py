#from __future__ import print_function

# This module is supposed to be imported first
# it contains global variables used for configuration


# try to register printing methods if IPython is running

try:
    import dolo.misc.printing as printing
    from numpy import ndarray
    from dolo.symbolic.model import Model

    ip = get_ipython()

    # there could be some kind of autodecovery there
    ip.display_formatter.formatters['text/html'].for_type( ndarray, printing.print_array )
    ip.display_formatter.formatters['text/html'].for_type( Model, printing.print_model )

    from IPython.core.display import display

except:

    from pprint import pprint
    def display(txt):
        pprint(txt)
