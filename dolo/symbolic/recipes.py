'''
ideas :
-  recursive blocks           [by default]
- (order left hand side ?)    [by default]
- dependency across blocks
- dummy blocks that are basically substituted everywhere else
'''

import os, yaml

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "recipes.yaml")

with file(DATA_PATH) as f:
  recipes = yaml.load(f)
