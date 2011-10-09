import argparse

from dolo import __version__

parser = argparse.ArgumentParser(description='RECS compiler')
parser.add_argument('-v','--version', action='version', version=__version__)
parser.add_argument('-s','--solve', action='store_const', const=True, default=False, help='solve for the decision rule')
parser.add_argument('input', help='model file')
parser.add_argument('output',help='model file')

args = parser.parse_args()

######

input_file = args.input
if args.output:
    output_file = args.output
else: # we should determine some good output name in case none has been specified
    pass

######

from dolo.misc.yamlfile import yaml_import

model = yaml_import( input_file )

from dolo.compiler.compiler_mirfac import MirFacCompiler

comp = MirFacCompiler(model)

if model['parameters_values']:
    with_parameters_values = True
else:
    with_parameters_values = False

txt = comp.process_output_matlab(target = 'recs',
                                 with_parameters_values = with_parameters_values,
                                 with_solution = args.solve)

######

with file(output_file,'w') as f:
    f.write(txt)
