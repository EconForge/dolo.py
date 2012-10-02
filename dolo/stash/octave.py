# -*- coding: utf-8 -*-
def send_to_octave(model,interactive=True,rmtemp=False,append=""):
    tempname = model.fname + ".mod"
    f = file(tempname,'w')
    modtext = model.export_to_modfile(options={},append=append)
    f.write(modtext)
    f.close()
    main_file = "%% main file for model : %s\n\n" % (model.fname)
    main_file += 'addpath("%s")\n' % ("/home/pablo/Sources/dynare_4_svn/matlab")
    main_file += 'dynare("%s")\n' % (model.fname)
    main_file += 'save -v6 %s M_ oo_ options_' % (model.fname + "_res.mat")   
    
    g = file( model.fname + "_main.m",'w')
    g.write(main_file)
    g.close()

    if interactive:
        import os
        command = 'gnome-terminal -e "octave --persist --eval \\"%s\\""' % ( model.fname + "_main.m" )
        os.system( command)
        return None
    else:
        import os
        import scipy
        from scipy import io
        command = 'octave --eval "%s"' % ( model.fname + "_main.m" )
        os.system(command)
        res = io.loadmat( model.fname + "_res.mat")
        return res
    
#def retrieve_results(mlab):
    #d_M = dict()
    #d_oo = dict()
    #d_options = dict()
  
    
    #m_options = mlab._get('options_')
    #m_names = mlab.fieldnames(m_options)
    #for m_name in m_names:
        ##print(m_name)
        #name = str(mlab.cell2mat(m_name))
        #print(name)
        ##name = m_name._[0]
        ##print(name)
        ##d_options[name] = getattr(m_options,name)#m_options.__getattr__(m_name)
    
    #return(d_options)