#! /usr/bin/python

# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="pablo"
__date__ ="$14 juin 2009 22:38:48$"

import sys
import os
import getopt
#import commands


from dolo import *
from dolo.compiler.compiler_dynare import *

if __name__ == "__main__":

    def main(argv):
        if len(argv)<1:
            print("Not enough arguments.")

            print('')

            usage()
            
            sys.exit(2)

        # Read options
        options = {"check":False,"ramsey":False, "dynare":False, "output":False, "static-order":2, "dynamic-order":1}
        short_arg_dict = "hcdro"
        long_arg_dict = ["help","check","dynare","static-order=","dynamic-order=","output=","ramsey"]
        try:
            opts, args = getopt.getopt(argv, short_arg_dict, long_arg_dict)
        except getopt.GetoptError:
            usage()
            sys.exit(2)
        for opt, arg in opts:
            if opt in ("-h","--help"):
                usage()
                sys.exit()
            if opt in ("-d","--dynare"):
                options["dynare"] = True
            if opt in ("-c","--check"):
                options["check"] = True
            if opt in ("-r","--ramsey"):
                options["ramsey"] = True
            if opt in ("-o","--output"):
                options["output"] = True
            if opt in ("--dynare",):
                options["dynare"]
            if opt in ("--static-order",):
                options["static-order"] = int(arg)
            if opt in ("--dynamic-order",):
                options["dynamic-order"] = int(arg)

        if args == []:
            print("File argument missing")
            sys.exit()
        else:
            filename = args[0]

        # determine filename type
        regex_mod = re.compile("(.*)\.mod")
        regex_mod_match = re.match(regex_mod,filename)
        if regex_mod_match:
            filetype = "mod"
            filename_trunc = regex_mod_match.groups()[0]

        regex_xml = re.compile("(.*)\.xml")
        regex_xml_match = re.match(regex_xml,filename)
        if regex_xml_match:
            filetype = "xml"
            filename_trunc = regex_xml_match.groups()[0]


        current_dir = os.getcwd()
        filename_trunc = current_dir + '/' + filename_trunc

        if filetype == "":
            print("Unknown filetype")
            sys.exit(2)

        # Import the model
        if filetype == "mod":
            dynare_model = dynare_import(filename)

        dynare_model.check(verbose=True)
        
        #elif filetype == "xml":
        #    import_xmlfile(filename_trunc,options)

        # Process the model
        if options['ramsey']:
            from dolo.symbolic.ramsey import RamseyModel
            #dynare_model.introduce_auxiliary_variables()
            dynare_model.check()
            
            rmodel = RamseyModel(dynare_model)
            rmodel.check(verbose=True)
            options['output'] = True
            dynare_model=rmodel


        if options['output']:
            from dolo.compiler.compiler_dynare import DynareCompiler
            comp = DynareCompiler(dynare_model)
            comp.export_to_modfile()

        if options['dynare']:
            from dolo.compiler.compiler_dynare import DynareCompiler
            model = dynare_model

            comp = DynareCompiler(model)

            write_file(model.fname + '_dynamic.m', comp.compute_dynamic_mfile(max_order=options["dynamic-order"]))
            write_file(model.fname + '_static.m', comp.compute_static_mfile(max_order=options["static-order"]))
            write_file(model.fname + '.m',  comp.compute_main_file() )
            

    def write_file(fname,content):
        f = file(fname,'w')
        f.write(content)
        f.close()
        
    def import_modfile(filename_trunc,options):
        from dolo.compiler.compiler_dynare import DynareCompiler
        import time

        t0 = time.time()
        '''read mod file ; write xml file'''
        #from misc.interactive import dynare_import
        filename = filename_trunc + ".mod"

        fname = re.compile('.*/(.*)').match(filename_trunc).group(1)
        
        model = dynare_import(filename)

        model.fname = fname

        #model = resp['model']

        model.check()

        comp = DynareCompiler(model)
        #comp.export_infos()
        

        #write_file(model.fname + '.m',  comp.compute_main_file() )
        print('Modfile preprocessing finished in {0} seconds'.format(time.time() - t0))
#        if options["check"]:
#            model.check_all(print_info=True,print_eq_info=True)
#        if options["portfolio"]:
#            from misc.portfolio import process_portfolio_model,process_portfolio_model_total
#            pf_solver = process_portfolio_model(solver)
#            print(pf_solver.export_to_modfile())


    def process_xmlfile(filename_trunc,options={}):
        '''process xml file ; returns parsed model with symbolic equations'''

        return(model)

    def write_ramsey_policy(filename_trunc,options,model):
        #model.export_to_modfile()
        ramsey_model = model.process_ramsey_policy_model()
        f = open(filename_trunc + "_ramsey.mod","w")
        f.write(ramsey_model.export_to_modfile())
        f.close()
        print("Ramsey model written to : " + filename_trunc + "_ramsey.mod")

    def write_portfolio_version(filename_trunc,options,model):
        print("Processing portfolios")
        portfolio_model = model.process_portfolio_model()
        f = open(filename_trunc + "_pf.mod","w")
        f.write(portfolio_model.export_to_modfile())
        f.close()
        print("Portfolio model written to : " + filename_trunc + "_pf.mod")

    def usage():
	help_text = '''
	Usage : dolo.py [options] model.xml|model.mod

        The file argument defining a model can be in Dynare format or in XML format. Its extension determines its type.
	Available options are :
				  	without option Dolo does nothing
	-h      --help            	print this message
        -c      --check           	model is checked for consistency

       	-r      --ramsey          	model's equations defining Ramsey's optimal policy are added (not implemented)

        -o      --outpout=OUTPUT  	processed model is written to OUTPUT file (not implemented)

        -d      --dynare
                --static-order=order    _static.m file is written at given order
		--dynamic-order=order	_dynamic.m file is written at given order
	'''
	print(help_text)

    main(sys.argv[1:])
