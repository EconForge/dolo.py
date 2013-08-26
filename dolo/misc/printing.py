from sympy.printing.str import StrPrinter


def sympy_to_dynare_string(sexpr):
    s = str(sexpr)
    s = s.replace("==","=")
    s = s.replace("**","^")
    return(s)


class DoloPrinter(StrPrinter):
    def _print_TSymbol(self,expr):
        return expr.__str__()

dp = DoloPrinter()



#############################################################################
# The following functions compute the HTML representation of common objects #
#############################################################################



def print_table( tab, col_names=None, row_names=None, header=False):
    #table_style = "background-color: #A8FFBE;"
    #td_style = "background-color: #F8FFBD; border: medium solid white; padding: 5px; text-align: right"
    txt_lines = ''
    for l in tab:
        txt_columns = ['\t\t<td>{0} </td>'.format(str(c)) for c in l]
        txt_columns = str.join('\n',txt_columns)
        txt_lines += '\t<tr>\n{0}\n</tr>\n'.format( txt_columns )
    txt = '<table>{0}</table>'.format(txt_lines)
    return txt

def print_array( obj,row_names=None,col_names=None):
    import numpy
    tab = numpy.atleast_2d( obj )
    resp = [[ "%.4f" %tab[i,j] for j in range(tab.shape[1]) ] for i in range(tab.shape[0]) ]
    if row_names:
        resp = [  [row_names[i]] + resp[i] for i in range(tab.shape[0]) ]
    if col_names:
        if row_names:
            resp = [[''] +col_names] + resp
        else:
            resp = [col_names] + resp
    return print_table(resp)

def print_model( model, print_residuals=True):

    from sympy import latex
    if print_residuals:
        from dolo.symbolic.model import compute_residuals
        res = compute_residuals(model)
    if len( model.equations_groups ) > 0:
        if print_residuals:
            eqs = [ ['', 'Equations','Residuals'] ]
        else:
            eqs = [ ['', 'Equations'] ]
        for groupname in model.equations_groups:
            eqg = model.equations_groups
            eqs.append( [ groupname ,''] )
            if print_residuals:
                eqs.extend([ ['','${}$'.format(latex(eq)),str(res[groupname][i])] for i,eq in enumerate(eqg[groupname]) ])
            else:
                eqs.extend([ ['','${}$'.format(latex(eq))] for eq in eqg[groupname] ])
        txt = print_table( eqs, header = True)
        return txt

    else:
        if print_residuals:
            table = [ (i+1, '${}$'.format(latex(eq)), str(res[i]) ) for i,eq in enumerate(model.equations)]
        else:
            table = [ (i+1, '${}$'.format(latex(eq)) ) for i,eq in enumerate(model.equations)]
        txt = print_table([['','Equations']] + table, header=True)
    return txt

def print_cmodel( cmodel, print_residuals=True):

    if cmodel.model is None:
        return str(cmodel)

    model = cmodel.model

    txt = 'Compiled model : {}\n'.format(model.name)
    txt += print_model(model, print_residuals=print_residuals)
    return txt

#from pandas import

def print_dynare_decision_rule( dr ):

    col_names = [v(-1) for v in dr.model.variables] + dr.model.shocks
    row_names = dr.model.variables

    #first_order_coeffs = df(column_stack([dr['g_a'],dr['g_e']]), columns = col_names, index = row_names)
    from numpy import column_stack
    [inds, state_vars] = zip( *[ [i,v] for i,v in enumerate(dr.model.variables) if v in dr.model.state_variables] )
    first_order_coeffs = column_stack([dr['g_a'][:,inds],dr['g_e']])

    col_names = [v(-1) for v in state_vars] + dr.model.shocks
    row_names = dr.model.variables
    txt = print_array(first_order_coeffs, col_names = col_names, row_names = row_names)
    return txt