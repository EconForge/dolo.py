#from __future__ import print_function

# This module is supposed to be imported first

# it contains global variables used for configuration

# platform :
#   - python
#   - sage

# engine


class MLabBridge:

    def __init__(self):
        self.engine = None
        
    def set_engine(self, engine_name):
        
        if engine_name == 'octave':
            import pytave as engine
            self.engine = engine
            self.engine_name = engine_name
        else:
            print('Unknown engine type : {0}'.format(engine_name))

    def __call__(self, cmd, nout=0):
        if self.engine_name == 'octave':
            resp = self.engine.eval(nout, cmd)
            return resp

    def feval(self, nargout, funcname, *arguments):
        if self.engine_name == 'octave':
            return self.engine.feval(nargout, funcname, *arguments)
            
    def dynare_config(self):
        try:
            self.__call__('dynare_version;') # how to print in a robust way ?
            dynare_version = self.__call__('version',1)[0]
            if self.engine_name == "octave":
                dynare_version = dynare_version[0].tostring()
            print( '- Dynare version is : ' + str(dynare_version) )
            self.__call__('dynare_config')
        except:
            print( '- Dynare is not in your matlab/octave path' )


class DefaultInterpreter():
    
    def display(self,obj):
        print(obj)

    display_html = display

class IPythonInterpreter(DefaultInterpreter):

    def __init__(self):
        from IPython.core.display import display, display_html
        import IPython
        v = IPython.__version__.split('.')
        if int(v[0]) == 0 and int(v[1]) < 11:
            raise(Exception('IPython is supported since version 0.11.'))
        self._display_ = display
        self.display_html = display_html

    def display(self,obj):
        if isinstance(obj,str):
            from dolo.misc.printing import HTMLString
            self._display_(HTMLString(obj))
        else:
            self._display_(obj)


use_engine = {
    'sylvester': False
}


#engine = MLabBridge()
#engine.set_engine('octave')
#engine("addpath('/home/pablo/Programmation/dynare/matlab/')")



save_plots = False


for IPC in [IPythonInterpreter, DefaultInterpreter]:
    try:
        interpreter = IPC()
        break
    except Exception as e:
        print e
        pass

del IPC

def display(obj):
    interpreter.display(obj)