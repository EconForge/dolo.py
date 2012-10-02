# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
from dolo.misc.modfile import parse_dynare_text

import os

DYNARE_MODFILES_PATH = 'examples/dynare_modfiles/'
exclude = [
    '.directory', # not a modfile

    't_lag2_check.mod', # 'periods=20;' is not understood
    't_lag2_checka.mod', # 'periods=20;' is not understood
    't_lag2a.mod', # 'periods=20;' is not understood
    't_lag2b.mod', # 'periods=20;' is not understood


    'variance_0.mod', # 'periods=20;' is not understood
    'test_matlab.mod', # 'periods=20;' is not understood    


    'ramsey.mod', # 'old-style shocks'
    'osr_example.mod', # 'old-style shocks'

    'ramst.mod', # 'deterministic shocks'
    'ramst_a.mod', # 'deterministic shocks'
    'ramst2.mod', # 'deterministic shocks'    


    't_periods_a.mod', # 'deterministic shocks'
    'example1_varexo_det.mod', # 'deterministic shocks'
    'ramst_normcdf.mod', # uses external function

    'predetermined_variables.mod', # predetermined variables

]

class  DynareModfileImportTestCase(unittest.TestCase):

    
    def test_dynare_modfile_import(self):
        # we test whether each of the modfile included with dynare
        # can be imported successfully

        modfiles = os.listdir(DYNARE_MODFILES_PATH)
        results = []
        for mf in [m for m in modfiles if m not in exclude]:
            fname = DYNARE_MODFILES_PATH + mf
            try:
                f = file(fname)
                txt  = f.read()
                f.close()
                model = parse_dynare_text(txt)
                res = (fname,'ok')
                results.append( res )
            except Exception as e:
                res = (fname, e)
                results.append( res )
            print res
        for r in results:
            print r

if __name__ == '__main__':
    unittest.main()

