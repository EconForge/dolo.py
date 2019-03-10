# -*- coding: utf-8 -*-
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

# # Buffer Stock Saving 
#
# In a model where impatient consumers face both transitory and permanent shocks _a la_ Friedman (1957), "buffer stock" saving arises: Consumers have a target wealth-to-permanent-income ratio, and actual wealth will be distributed around the target.
#
# [This paper](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#The-Problem) defines a buffer stock saving model and calibrates parameters:
#
# | Parameter | Description | Code | Value |
# | :---: | ---         | ---  | :---: |
# | $\newcommand{\PermGroFac}{\Gamma}\PermGroFac$ | Permanent Income Growth Factor | $\texttt{PermGroFac}$ | 1.03 |
# | $\newcommand{\Rfree}{\mathrm{\mathsf{R}}}\Rfree$ | Interest Factor | $\texttt{Rfree}$ | 1.04 |
# | $\newcommand{\DiscFac}{\beta}\DiscFac$ | Time Preference Factor | $\texttt{DiscFac}$ | 0.96 |
# | $\newcommand{\CRRA}{\rho}\CRRA$ | Coeﬃcient of Relative Risk Aversion| $\texttt{CRRA}$ | 2 |
# | $\newcommand{\UnempPrb}{\wp}\UnempPrb$ | Probability of Unemployment | $\texttt{UnempPrb}$ | 0.005 |
# | $\newcommand{\IncUnemp}{\mu}\IncUnemp$ | Income when Unemployed | $\texttt{IncUnemp}$ | 0. |
# | $\newcommand{\PermShkStd}{\sigma_\psi}\PermShkStd$ | Std Dev of Log Permanent Shock| $\texttt{PermShkStd}$ | 0.1 |
# | $\newcommand{\TranShkStd}{\sigma_\theta}\TranShkStd$ | Std Dev of Log Transitory Shock| $\texttt{TranShkStd}$ | 0.1 |
#
# For a microeconomic consumer with 'Market Resources' (net worth plus current income) $M_{t}$, end-of-period assets $A_{t}$ will be the amount remaining after consumption of $C_{t}$.  <!-- Next period's 'Balances' $B_{t+1}$ reflect this period's $A_{t}$ augmented by return factor $R$:-->
# \begin{eqnarray}
# A_{t}   &=&M_{t}-C_{t}  \label{eq:DBCparts} \\
# # %B_{t+1}   & = & A_{t} R \notag \\
# \end{eqnarray}
#
# The consumer's permanent noncapital income $P$ grows by a predictable factor $\PermGroFac$ and is subject to an unpredictable lognormally distributed multiplicative shock $\mathbb{E}_{t}[\psi_{t+1}]=1$, 
# \begin{eqnarray}
# P_{t+1} & = & P_{t} \PermGroFac \psi_{t+1}
# \end{eqnarray}
#
# and actual income is permanent income multiplied by a logormal multiplicative transitory shock, $\mathbb{E}_{t}[\theta_{t+1}]=1$, so that next period's market resources are
# \begin{eqnarray}
# # %M_{t+1} &=& B_{t+1} +P_{t+1}\theta_{t+1},  \notag
# M_{t+1} &=& A_{t}R +P_{t+1}\theta_{t+1}.  \notag
# \end{eqnarray}
#
# When the consumer has a CRRA utility function $u(c)=\frac{c^{1-\rho}}{1-\rho}$, the paper shows that the problem can be written in terms of ratios of money variables to permanent income, e.g. $m_{t} \equiv M_{t}/P_{t}$, and the Bellman form of [the problem reduces to](http://econ.jhu.edu/people/ccarroll/papers/BufferStockTheory/#The-Related-Problem):
#
# \begin{eqnarray*}
# v_t(m_t) &=& \max_{c_t}~~ u(c_t) + \beta~\mathbb{E}_{t} [(\Gamma\psi_{t+1})^{1-\rho} v_{t+1}(m_{t+1}) ] \\
# & s.t. & \\
# a_t &=& m_t - c_t \\
# m_{t+1} &=& R/(\Gamma \psi_{t+1}) a_t + \theta_{t+1} \\
# \end{eqnarray*}
#
# and the Euler equation for this model is 
#
# \begin{eqnarray*}
# c_{t}^{-\rho} & = & R \beta \mathbb{E}_{t}[(\Gamma \psi c_{t+1})^{-\rho})] \\
# 0 & = & R \beta \mathbb{E}_{t}[(\Gamma \psi c_{t+1}/c_{t})^{-\rho})]-1
# \end{eqnarray*}
#
#
# For the purposes of this notebook, the paper's baseline parameterization is changed as follows:
#
# 1. The unemployment (zero-income event) shocks are turned off
# 2. An explicit liqudity constraint is added ($c_{t} \leq m_{t}$)

from dolo import *
from matplotlib import pyplot as plt 

model = yaml_import("../models/bufferstock.yaml")

print ( model )

# +
# Set a maximum range for market resources and solve for the model's decision rule
model.data['calibration']['max_m'] = 10.0
model.data['domain']['m'] = [0,'max_m']

dr = time_iteration(model)
# -

# ## The Ergodic distribution is roughly centered around the target

sim_long = simulate(model, dr, i0=1, N=1000, T=200)

import seaborn
seaborn.distplot(sim_long.sel(T=199, V='m'))
plt.xlabel("$m$")

# ## Plotting the Decision Rule

tab = tabulate(model, dr,'m')

from matplotlib import pyplot as plt

stable_wealth = model.eval_formula('1/R+(1-1/R)*m(0)', tab)
plt.plot(tab['m'], tab['m'],color='black', linestyle='--')
plt.plot(tab['m'], stable_wealth,color='black', linestyle='--')
plt.plot(tab['m'], tab['c'])
plt.xlabel("$m_t$")
plt.ylabel("$c_t$")
plt.grid()
