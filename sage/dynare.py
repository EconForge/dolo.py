# -*- coding: utf-8 -*-
import time
import inspect

# here we clean up the sage workspace (highly unoptimal)
try:
    del dynkin_diagram
except:
    None

def set_matlab_engine(engine):
    '''Set the interface to be used for computations. Engine can be :
    - 'octave'
    - 'matlab' (needs matlab installed)
    - 'matlabw' (needs matlab and mlabwrap installed)
    '''
    if engine == 'octave':
        engine = octave
    elif engine == 'matlab':
        engine = matlab
    else:
        raise Exception('Non supported engine ' + str(engine) )
    globals()['engine'] = engine
    print( '- Calculation engine is : ' + str(engine) + ' ' + engine.version() )
    return None

set_matlab_engine('octave')

engine.execute('cd ' + DATA)

try:
    dynare_version = engine.execute('dynare_version').split('=')[1].strip()
    print( '- Dynare version is : ' + dynare_version )
except:
    print( '- Dynare is not in your matlab/octave path' )
    


try:
    import dolo
    _dolo_available = True
except ImportError:
    print( '- Dolo library not available.' )
    _dolo_available = False


def dyn_help():
    print( 'Help system not available yet.' )
    print( "Type 'dyn_help_commands' to get a list of all available commands." )
    print( '' )
    print( "Write 'dyn' and press Tab to get all dynare functions.") 

def dyn_help_commands():
    print( 'Available commands : ' )
    print( '- dynare( fname ) : runs Dynare calculation on specified modfile.')
    print( 'Dolo commands : ')
    print( '- print_model() : prints current model (must be defined in a modfile cell).' )
    print( '- compute_residuals() : returns residuals of all equations.')

print "- Type 'dyn_help()' for help."


def clear_data_directory():
    '''Remove all files from the DATA directory'''
    return None

def retrieve_from_matlab(name,mlab=engine,tname=None):
    import scipy.io as matio
    pwd = str(mlab.pwd()).strip()
    tname = name + '_' + str(time.time())
    cmd = "save('{0}','{1}','-v7')".format(tname + '.mat',name)
    mlab.execute(cmd)
    fname = pwd + '/' + tname + '.mat'
    s = matio.loadmat(fname,struct_as_record=True)
    s = s[name]
    os.remove(fname)
    return s

def get_current_model():
    try:
        model = globals()['dynare_model']
        return model
    except:
        raise( Exception( "You need to define a model first !" ) )

def compute_residuals(model=None):
    if model == None:
        model = get_current_model()
    from dolo.misc.calculus import solve_triangular_system
    dvars = dict()
    dvars.update(model.parameters_values)
    dvars.update(model.init_values)
    for v in model.variables:
        if v not in dvars:
            dvars[v] = 0
    # what are we supposed to do with parameters ?
    values = solve_triangular_system(dvars)[0]
    stateq = [ eq.subs( dict([[v,v.P] for v in eq.variables]) ) for eq in model.equations]
    stateq = [ eq.subs( dict([[v,0] for v in eq.shocks]) ) for eq in stateq]
    stateq = [ eq.rhs - eq.lhs for eq in stateq ]
    residuals = [ eq.subs(values) for eq in stateq ]
    return residuals
        
def print_model(model=None, print_residuals=True, print_names=False):
    if model == None:
        model = get_current_model()
    if print_residuals:
        res = compute_residuals(model)
        html.table([(i+1,model.equations[i],"%.4f" %float(res[i])) for i in range(len(model.equations))])
    elif print_names:
        html.table([(eq.n,eq,eq.name) for eq in model.equations])
    else:
        html.table([(i+1,model.equations[i]) for i in range(len(model.equations))])

class ModFileCell():

    def __init(self):
        self.fname = 'anonymous'

    def __call__(self,**args):
        self.fname = args.get('fname') if args.get('fname')  else 'anonymous'
        return self

    def eval(self,s,d,locals={}):
        s = s.encode() # this is to avoid incompatibilities if cell contains unicode
        #locals['dynare_modfile'] = s
        DATA = locals['DATA']
        mlab = locals['engine']

        fname = self.fname

        #try:
        from dolo.misc.interactive import parse_dynare_text

        t = parse_dynare_text(s,full_output=True)
        t['model'].fname = fname
        locals['dynare_model'] = t['model']
        locals['modfile_content'] = t
        print 'Dynare model successfully parsed and stored in "dynare_model"'
        locals['dynare_model'].check()
#        except NameError:
#            print 'Dolo is not installed'
#            None
            # do nothing : Dolo is not installed
        #except Exception as exc:
        #    print 'Dolo was unable to parse dynare block.'
        #    print exc

        f = file(DATA + fname + '.mod' ,'w')
        f.write(s)
        f.close()
        print "Modfile has been written as : '{0}.mod'".format(fname)

        return ''

def dynare(fname,*kargs):
    '''Calls dynare() with arguments supplied'''
    args = [fname] + list(kargs)
    cmd = "dynare({0})".format(','.join(["'%s'" %o for o in args]) )
    print engine.execute( cmd )

modfile = ModFileCell()
