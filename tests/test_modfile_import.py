# To change this template, choose Tools | Templates
# and open the template in the editor.

import unittest
from dolo.misc.interactive import parse_dynare_text

import os

DYNARE_MODFILES_PATH = '../examples/dynare_modfiles/'

class  DynareModfileImportTestCase(unittest.TestCase):
    
    def test_dynare_modfile_import(self):
        # we test whether each of the modfile included with dynare
        # can be imported successfully

        modfiles = os.listdir(DYNARE_MODFILES_PATH)
        for mf in modfiles:
            fname = DYNARE_MODFILES_PATH + mf
            print 'Trying to import : ' + mf
            f = file(fname)
            txt  = f.read()
            f.close()
            resp = parse_dynare_text(txt)


if __name__ == '__main__':
    unittest.main()

