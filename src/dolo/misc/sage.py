
def show_file(f):
    import os
    fname = os.path.basename(f)
    os.symlink(f, fname)

#import os
#
#class MatlabEngine:
#    # Very preliminary implementation
#    def __init__(self, engine, DATA):
#        self.engine = engine
#        self.engine_name = engine.name()
#        globals()['DATA'] = DATA
#
#        #self.set_matlab_engine(engine)
#
#    def set_matlab_engine(self,engine_name):
#        if engine_name == 'octave':
#            engine = octave
#        elif engine_name == 'matlab':
#            engine = matlab
#        elif engine_name == 'mlabwrap':
#            from mlabwrap import mlab
#            engine = mlab
#            mlab._autosync_dirs = False
#        else:
#            raise Exception('Non supported engine ' + str(engine) )
#        print( '- Calculation engine is : ' + str(engine_name) + ' - ' + engine.version() )
#        self.engine = engine
#        self.engine_name = engine_name
#        self.execute('cd ' + DATA)
#        #globals()['engine'] = self
#
#    def execute(self,cmd):
#        files_before = os.listdir( DATA )
#        files_before_mtime = [ os.lstat(DATA + e).st_mtime for e in files_before]
#        if self.engine_name in ('matlab','octave'):
#            output = self.engine.execute(cmd)
#        elif self.engine_name == 'mlabwrap':
#            output = self.engine._do(cmd, nout=0)
#        files_after = os.listdir( DATA )
#        files_after_mtime = [ os.lstat(DATA + e).st_mtime for e in files_after]
#        modified = [ f for f in files_after if ( f not in files_before ) or ( files_after_mtime[files_after.index(f)] > files_before_mtime[files_before.index(f)] ) ]
#        for f in modified:
#            os.symlink(DATA+f,'./'+f)
#        return output
#
#    def eval(self, code, strip=True, synchronize=False, locals=None, **kwargs):
#        files_before = os.listdir( DATA )
#        files_before_mtime = [ os.lstat(DATA + e).st_mtime for e in files_before]
#        instructions = code.split()
#        for ins in instructions:
#            print self.execute( ins )
#        files_after = os.listdir( DATA )
#        files_after_mtime = [ os.lstat(DATA + e).st_mtime for e in files_after]
#        modified = [ f for f in files_after if ( f not in files_before ) or ( files_after_mtime[files_after.index(f)] > files_before_mtime[files_before.index(f)] ) ]
#        for f in modified:
#            os.symlink(DATA+f,'./'+f)
#
#    def set(self, var,value):
#        import scipy
#        import tempfile
#        import os
#        import time
#        if isinstance(value,np.ndarray):
#            tmpname = '/tmp/dolo_' + str(time.time())
#            scipy.io.savemat(tmpname + '.mat', {var:value})
#            self.execute('load {0}.mat'.format(tmpname) )
#            os.remove(tmpname + '.mat')
#            return None
#        else:
#            raise Exception("I don't know what to do with this value")
#
#    def get(self,var):
#        import scipy.io as matio
#        import tempfile
#        tmpname = '/tmp/dolo_' + str(time.time())
#        cmd = "save('{0}','{1}','-v7');".format(tmpname +'.mat',var)
#        self.execute(cmd)
#        s = matio.loadmat(tmpname+'.mat',struct_as_record=True)
#        s = s[var]
#        os.remove(tmpname+'.mat')
#        return s