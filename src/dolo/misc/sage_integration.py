# -*- coding: utf-8 -*-
from interactive import parse_dynare_text


class ModFileCell:

    def __init(self):
        self.fname = 'anonymous'

    def __call__(self,**args):
        self.fname = args.get('fname') if args.get('fname')  else 'anonymous'
        return self

    def eval(self,s,d,locals={}):
        s = s.encode() # this is to avoid incompatibilities if cell contains unicode
        locals['dynare_modfile'] = s
        DATA = locals['DATA']
        mlab = locals['matlab']

        fname = self.fname

        try:
            t = parse_dynare_text(s,full_output=True)
            t['model'].fname = fname
            locals['dynare_model'] = t
            print 'Dynare block successfully parsed and stored in "dynare_model"'
        except Exception as exc:
            print 'Dolo was unable to parse dynare block.'
            print exc

        f = file(DATA + fname + '.mod' ,'w')
        f.write(s)
        f.close()
        print "Modfile has been written as : '{0}.mod'".format(fname)
#        if not mlab.is_running():
#            print "Starting matlab"
#        mlab.execute('cd ' + DATA)
        #mlab.execute("dynare('{0}'.mod)".format(fname))
        return None

class TextFileCell:

    def __call__(self,fname,**args):
        self.fname = fname
        return self

    def eval(self,s,d,locals={}):
        if not self.fname:
            raise Exception("File name not specified. Write : textfile('filename')")
        
        s = s.encode() # this is to avoid incompatibilities if cell contains unicode

        DATA = locals['DATA']

        fname = self.fname

        f = file(DATA + fname  ,'w')
        f.write(s)
        f.close()

        return None

modfile = ModFileCell()

textfile = TextFileCell()
