# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.1'
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

# %% [markdown]
# # Debug simple BufferStock model

# %%
from dolo import *
from matplotlib import pyplot as plt

# %% [markdown]
# ## The problem

# %%
model = yaml_import("../models/bufferstock.yaml")

# %%
dr0 = time_iteration(model)

# %% [markdown]
# The second column which measures successive approximations errors wanders in the greater than 1 region, which shows errors are not always decreasing. That is suspicious. Also, in the ast iterations the last column shows that the problem is not solved exactly in each iteration (newton iterations are capped at 10). Let's look at the solution.

# %%
tab = tabulate(model, dr0, 'm')
plt.plot(tab['m'],tab['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])

# %% [markdown]
# ## Solution 1

# %% [markdown]
# Hmmm... decision rule is identically equal to 0. It's simple and elegant and probably solves the euler equations exactly. Looks like it is also stable by bacward time-iteration. Why not keep it?

# %% [markdown]
# ## Solution 2
#
# Maybe the domain is too big. One can add a parameter `max_m` in the calibratoin section and make the domain depend on it.
# This can be done in the yaml file, or equivalently, directly in the data attribute (at your own risk).

# %%
model = yaml_import('bufferstock_IID.yaml')

# %%
model.data['calibration']['max_m'] = 10.0
model.data['domain']['m'] = [0,'max_m']

# %%
σ_perm = model.get_calibration()['σ_perm']
μ_perm = np.exp((σ_perm ** 2)/2)
μ_perm
σ_tran = model.get_calibration()['σ_tran']
μ_tran = np.exp((σ_tran ** 2)/2)
μ_tran

# %%
model.exogenous.discretize().integration_nodes[:,0]

# %%
np.transpose(model.exogenous.discretize().integration_nodes[:,1])

# %%
np.dot(np.exp(model.exogenous.discretize().integration_nodes[:,1]),model.exogenous.discretize().integration_weights)

# %%
np.dot(model.exogenous.discretize().integration_weights,model.exogenous.discretize().integration_nodes)

# %%
model.exogenous.discretize().integration_weights

# %% [markdown]
# Then we first try to solve the model on a smaller domain, before increasing its size.

# %%
model.set_calibration(max_m=2)

# %%
dr1 = time_iteration(model)

# %%
tab = tabulate(model, dr1, 'm')
plt.plot(tab['m'],tab['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])

# %% [markdown]
# This seems to work. Let's increase the domain and take this result as initial guess.

# %%
model.set_calibration(max_m=10)
dr2 = time_iteration(model, initial_guess=dr1)

# %%
tab = tabulate(model, dr2, 'm')
plt.plot(tab['m'],tab['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])

# %% [markdown]
# This looks more like the solution. We can get it faster using the  `improved_time_iteration` algorithm. This is undocumented, and taken from my paper "Back in Time. Fast".

# %%
model.set_calibration(max_m=5) # small state-space
dr1_iti = improved_time_iteration(model, invmethod='gmres',verbose=True)
model.set_calibration(max_m=10) # small state-space
dr2_iti = improved_time_iteration(model, initial_dr=dr1_iti, invmethod='gmres',verbose=True)

# %%
tab_iti = tabulate(model, dr1_iti, 'm')
plt.plot(tab_iti['m'],tab_iti['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])
plt.plot(tab_iti['m'],tab_iti['c'])

# %% [markdown]
# ## Solution 3:  change initial guess
#
# The initial time iteration, suggests the initial guess is too far from the solution. Let's try to change it.

# %%
model = yaml_import('bufferstock.yaml')

# %% [markdown]
# Let's try a first order decision rule.

# %%
drp = approximate_controls(model)

# %%
dr1 = time_iteration(model, initial_guess=drp)

# %%
tab = tabulate(model, dr1, 'm')
plt.plot(tab['m'],tab['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])

# %% [markdown]
# This is still arguably the wrong solution. Lets's try to understand what's happening by recordining the successive decision rules.
# There are some plans to add this type of diagnostics information in Dolo. For now, it can be done by hacking into the time_iteration function using the hook convenience function. This method is very flexible and doesn't require to clutter the time-iteration code with lot's of optional branches.

# %%
import inspect
import copy

record = []
def record_dr():
    """This function is called at each iteration and looks at its surrounding to record decision rules at each iteration."""
    frame = inspect.currentframe().f_back
    dr = frame.f_locals['mdr']
    it = frame.f_locals['it']
    record.append((it,copy.copy(dr)))

# %%
time_iteration(model, hook=record_dr, verbose=False)

# %%
# we convert each recorded dr into a dataframe to be plotted
tabs = [tabulate(model, dr, 'm') for it,dr in record]

# %%
plt.plot(tabs[0]['m'],tabs[0]['m'],linestyle='--',color='black')
# this one approximates, 
plt.plot(tabs[0]['m'],1+0*tabs[0]['m'],linestyle='--',color='black')
for i,t in enumerate(tabs):
    if i%10==0:
        plt.plot(t['m'],t['c'], alpha=0.1, color='red')
plt.ylim(0,2)

# %% [markdown]
# What happens in each iteration, even the very first ones, is plenty clear. The initial decision rule is a constant c=0.9 doesn't have a fixed point, and looks like none of the subsequent iterations does. Maybe if we choose a higher constant as an initial guess ?

# %%
model.set_calibration(c=1.5)

# %%
dr = time_iteration(model)

# %%
tab = tabulate(model, dr, 'm')
plt.plot(tab['m'],tab['m'],linestyle=':',color='grey')
plt.plot(tab['m'],tab['c'])
plt.grid()

# %% [markdown]
# This works!

# %%
%time dr_iti = improved_time_iteration(model)

# %% [markdown]
# This does too!

# %% [markdown]
# Remark: it is also possible to supply any other initial decision rule, as simple rule, by supplying as initial guess an object which implements the method `.eval_is`

# %%


