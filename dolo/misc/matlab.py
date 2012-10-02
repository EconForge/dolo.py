from dolo.symbolic.symbolic import Variable,Parameter

import sympy

def convert_struct_to_dict(s):
    # imperfect but enough for now
    if len(s.dtype) == 0:
        if s.shape == (1,):
            return str(s[0])
        elif s.shape == (0,0):
            return []
        elif s.shape == (1,1):
            return s[0,0]
        else:
            return s
    else:
        # we suppose that we found a structure
        d = dict()
        ss = s[0,0] # actual content of the structure
        for n in ss.dtype.names:
            d[n] = convert_struct_to_dict(ss[n])
        return d

def send_to_matlab(model,interactive=True,rmtemp=False,append=""):
    tempname = model.fname + ".mod"
    f = file(tempname,'w')
    modtext = model.export_to_modfile(options={},append=append)
    f.write(modtext)
    f.close()
    main_file = "%% main file for model : %s\n\n" % (model.fname)
    main_file += "addpath('%s')\n" % ("/home/pablo/Sources/dynare_v4/matlab")
    main_file += "dynare('%s')\n" % (model.fname)
    main_file += "save -v6 %s M_ oo_ options_\n" % (model.fname + "_res.mat")
    main_file += "exit"
    
    g = file( model.fname + "_main.m",'w')
    g.write(main_file)
    g.close()

    if interactive:
        pass
        #import os
        #command = 'gnome-terminal -e "matlab --persist --nodesktop --nojvm --eval \\"%s\\""' % ( model.fname + "_main.m" )
        #os.system( command)
        #return None
    else:
        import os
        import scipy
        from scipy import io
        command = "matlab -nodesktop -nojvm -r %s" % ( model.fname + "_main" )
        os.system(command)
        res = io.loadmat( model.fname + "_res.mat")
        return res


def send_to_dynare_2(solver,mlab,rmtemp=False):
    tempname = solver.model.fname + ".mod"
    f = file(tempname,'w')
    modtext = solver.export_to_modfile(options={},append="")
    f.write(modtext)
    mlab.dynare(solver.model.fname)
    f.close()

def retrieve_results(mlab):
    d_options = dict()
    m_options = mlab._get('options_')
    m_names = mlab.fieldnames(m_options)
    for m_name in m_names:
        #print(m_name)
        name = str(mlab.cell2mat(m_name))
        print(name)
        #name = m_name._[0]
        #print(name)
        #d_options[name] = getattr(m_options,name)#m_options.__getattr__(m_name)
    return(d_options)

def value_to_mat(v):
    if isinstance(v,bool):
        if v:
            return '1'
        else:
            return '0'
    elif ( isinstance(v,float) or isinstance(v,int) ):
        return str(v)
    elif isinstance(v,list):
        l = [value_to_mat(i) for i in v]
        classes = [i.__class__ for i in v]
        if (str in classes) or (sympy.Symbol in classes) or (Parameter in classes) or (Variable in classes): #list contains at least one string
            return 'strvcat(%s)' %str.join(' , ',l)
        else:
            return '[%s]' %str.join(' ; ',l)
    elif isinstance(v,sympy.Matrix):
        return '[%s]' %v.__repr__().replace('\n',';').replace(',',' ')
    elif str(v.__class__) == "<type 'numpy.ndarray'>":
        if len(v.shape)  <= 2:
            return str(v).replace('\n','').replace('] [',' ; ')
        else:
            import numpy
            return 'reshape( {0} , {1} )'.format(  str(v.flatten('F')).replace('\n','') , str(v.shape).strip('()')  )
        #raise Warning('list conversion to matlab not implemented (will be soon)')
    else:
        return "'%s'" %str(v)


def struct_to_mat(d,name = 'dict'):
    # d must be a dictionary
    txt = '%s = struct;\n' %name
    for k in d.keys():
        if isinstance(k,float) or isinstance(k,int):
            key = 'n%s' % str(k)
        else:
            key = str(k)
        if isinstance(d[k],dict):
            txt += struct_to_mat(d[k],key)
            txt += "%s = setfield(%s,'%s',%s) ;\n" %(name, name , key, key)
            #raise(Exception('recursive dictionaries not allowed yet'))
        else:
            v = value_to_mat(d[k])
            txt += "%s = setfield(%s,'%s',%s) ;\n" %(name, name , key, v)
    return(txt)