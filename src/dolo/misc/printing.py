from sympy.printing.str import StrPrinter

class DoloPrinter(StrPrinter):
    def _print_TSymbol(self,expr):
        return expr.__str__()

dp = DoloPrinter()



#############################################################################
# The following functions compute the HTML representation of common objects #
#############################################################################

import dolo

class HTMLString(str):
    def _repr_html_(self):
        return self


class HTMLPrinter():

    def __call__(self):
        return self # for compatibility purposes
    
#    def doprint(self,obj):
#        if isinstance(obj,list):
#            # test whether it is a table
#            if isinstance(obj[0],list):
#                ne = len(obj[0])
#                test = [ isinstance(e,(list,tuple)) and len(e) == ne for e in obj]
#                if False not in test:
#                    return self.print_table(obj)
#        elif isinstance(obj, dolo.symbolic.model.Model):
#            return self.print_model(obj)
#        else:
#            return HTMLString(str(obj))

    def print_table(self, tab, col_names=None, row_names=None, header=False):
        txt_lines = ''
        for l in tab:
            txt_columns = ['<td>{0}</td>'.format(str(c)) for c in l]
            txt_columns = str.join('\n',txt_columns)
            txt_lines += '<tr>{0}</tr>'.format( txt_columns )
        txt = '<table border = 1 bgcolor="#F8FFBD">{0}</table>'.format(txt_lines)
        return HTMLString(txt)

    def print_array(self,obj):
        tab = obj
        resp = [[ "%.3f" %tab[i,j] for j in range(tab.shape[1]) ] for i in range(tab.shape[0]) ]
        if row_names:
            resp = [  [row_names[i]] + resp[i] for i in range(tab.shape[0]) ]
        header = False
        if col_names:
            resp = [[''] +col_names] + resp
            header = True
        return self.print_table(resp,header)

    def print_model(self,model, print_residuals=True):
        if print_residuals:
            from dolo.symbolic.model import compute_residuals
            res = compute_residuals(model)
            txt = self.print_table([['','Equations','Residuals']] + [(i+1,model.equations[i],"%.4f" %float(res[i])) for i in range(len(model.equations))],header=True)
        else:
            txt = self.print_table([['','Equations']] + [(i+1,model.equations[i]) for i in range(len(model.equations))], header=True)
        return HTMLString(txt)

htmlprinter = HTMLPrinter()