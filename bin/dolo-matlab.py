#!/usr/bin/python

import argparse

from dolo import __version__

parser = argparse.ArgumentParser(description='Matlab compiler')
parser.add_argument('-v','--version', action='version', version=__version__)
parser.add_argument('-r','--print_residuals', action='store_const', const=True, default=False, help='print residuals at the steady-state')
parser.add_argument('-s','--solve', action='store_const', const=True, default=False, help='solve for the decision rule')
parser.add_argument('-o','--order', nargs=1, type=int, default=[1], help='solution order (1,2,3)')
parser.add_argument('input', help='model file')
parser.add_argument('output',nargs='?',type=str,default=None,help='model file')

args = parser.parse_args()

######

input_file = args.input
if args.output:
    output_filename = args.output
    output_rad = output_filename.strip('.m')
else: # we should determine some good output name in case none has been specified
    output_rad = input_file.strip('.yaml') + '_model'
    output_filename = output_rad + '.m'

######

from dolo.misc.yamlfile import yaml_import

model = yaml_import( input_file )

import os
import re

basename = os.path.basename(output_filename)
fname = re.compile('(.*)\.m').match(basename).group(1)
model['name'] = fname


# check steady-state
if args.print_residuals:
    from dolo.symbolic.model import print_residuals
    print_residuals(model)


from dolo.compiler.compiler_matlab import CompilerMatlab
comp = CompilerMatlab(model)

if args.solve:
    solution_order = args.order[0]
else:
    solution_order = None
txt = comp.process_output( solution_order=solution_order, fname=output_rad)

######

with file(output_filename,'w') as f:
    f.write(txt)