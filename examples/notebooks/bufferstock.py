# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.6.7
#   varInspector:
#     cols:
#       lenName: 16
#       lenType: 16
#       lenVar: 40
#     kernels_config:
#       python:
#         delete_cmd_postfix: ''
#         delete_cmd_prefix: 'del '
#         library: var_list.py
#         varRefreshCmd: print(var_dic_list())
#       r:
#         delete_cmd_postfix: ') '
#         delete_cmd_prefix: rm(
#         library: var_list.r
#         varRefreshCmd: 'cat(var_dic_list()) '
#     types_to_exclude:
#     - module
#     - function
#     - builtin_function_or_method
#     - instance
#     - _Feature
#     window_display: false
# ---

from dolo import *
from matplotlib import pyplot as plt 

model = yaml_import("../models/bufferstock.yaml")

dr = time_iteration(model)

# One can also try the faster version

dr

# ## Stochastic Simulations

# Shocks are discretized as a markov chain by default:
dp = model.exogenous.discretize()
sim_shock = dp.simulate(10, 100, i0=1)
for i in range(10):
    plt.plot(sim_shock[:,i,0], color='red', alpha=0.5)

sim = simulate(model, dr, i0=1, N=100)

# +
plt.subplot(121)
for i in range(10):
    plt.plot(sim.sel(N=i,V='c'), color='red', alpha=0.5)
plt.ylabel("$c_t$")
plt.xlabel("$t$")
plt.subplot(122)
for i in range(10):
    plt.plot(sim.sel(N=i,V='m'), color='red', alpha=0.5)
plt.xlabel("$t$")
plt.ylabel("$w_t$")

plt.tight_layout()
# -

# ## Ergodic distribution

sim_long = simulate(model, dr, i0=1, N=1000, T=200)

import seaborn
seaborn.distplot(sim_long.sel(T=199, V='m'))
plt.xlabel("$m$")

# ## Plotting Decision Rule

tab = tabulate(model, dr,'m')

from matplotlib import pyplot as plt

stable_wealth = model.eval_formula('1/R+(1-1/R)*m(0)', tab)
plt.plot(tab['m'], tab['m'],color='black', linestyle='--')
plt.plot(tab['m'], stable_wealth,color='black', linestyle='--')
plt.plot(tab['m'], tab['c'])
plt.xlabel("m_t")
plt.ylabel("c_t")
plt.grid()
