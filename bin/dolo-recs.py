#!/usr/bin/python

import argparse
import os
import re
from dolo import __version__
from dolo.misc.yamlfile import yaml_import
from dolo.compiler.compiler_mirfac import MirFacCompiler

# Parse input arguments
parser = argparse.ArgumentParser(description='RECS compiler')
parser.add_argument('-v', '--version', action='version', version=__version__)
parser.add_argument('input', help='model file')
parser.add_argument('output', nargs='?', type=str, default=None, help='model file')

args = parser.parse_args()

input_file = args.input

# Output name
if args.output:
    output_filename = args.output
    output_rad      = output_filename.strip('.m')
else:
    output_rad      = input_file.strip('.yaml') + 'model'
    output_filename = output_rad + '.m'

# Parse yaml file
model = yaml_import(input_file)

# Model name based on output file name
basename      = os.path.basename(output_filename)
fname         = re.compile('(.*)\.m').match(basename).group(1)
model['name'] = fname

# Compilation for Matlab
comp = MirFacCompiler(model)
txt  = comp.process_output_recs(solution_order=None, fname=output_rad)

# Write output
with file(output_filename,'w') as f:
    f.write(txt)
