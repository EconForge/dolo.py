import pandas as pd

import numpy as np



default_layout = {
    1: [(4,4),(1,1)],
    2: [(6,4),(1,2)],
    3: [(8,4),(1,3)],
    4: [(8,8),(2,2)],
    5: [(8,8),(2,3)],
    6: [(8,8),(2,3)],
    7: [(6,8),(4,2)],
    8: [(6,8),(4,2)],
    9: [(8,8),(3,3)],
    10: [(12,8),(4,3)],
    11: [(12,8),(4,3)],
    12: [(12,8),(4,3)]
}

def plot_irfs(sims, variables=None, titles=None, layout=None, horizon=None, figsize=None, plot_options={}, line_options=None):

    from matplotlib import pyplot as plt

    if not isinstance(sims, list):
        # we have only one series to compare
        sims = [sims]

    if variables is None:
        variables = sims[0].columns

    if horizon is None:
        horizon = sims[0].shape[0]

    if titles is None:
        titles = variables

    n_v = len(variables)
    max_v = max(default_layout.keys())
    if n_v>max_v:
        raise Exception("Too many graphs")

    if layout is None:
        if figsize is None:
            figsize, layout = default_layout[n_v]
        else:
            layout = default_layout[n_v][1]

    if line_options is None:
        line_options = [{}]*n_v


    fig, axes = plt.subplots(layout[0], layout[1], figsize=figsize)
    axes = axes.ravel()
    for i,v in enumerate(variables):
        ax = axes[i]
        for j, sim in enumerate(sims):
            arguments = {}
            arguments.update(plot_options)
            arguments.update(line_options[j])
            ax.plot(sim[v][:horizon], **arguments)
        ax.set_title(titles[i])
        ax.set_xlim(0,horizon)
        max_ = max( [sim[v].max() for sim in sims] )
        min_ = min( [sim[v].min() for sim in sims] )
        if (max_-min_)>0.001: # pretty arbitrary
            margin = (max_-min_)*0.05 # 5% on each side
            ax.set_ylim(min_-margin, max_+margin)
        ax.grid(True)
    # fig.tight_layout()

    return fig



if __name__ == "__main__":

    A = np.random.random((50, N))
    A *= 0.9**(np.arange(N))[None,:]

    B = np.random.random((50, N))
    B *= 0.9**(np.arange(N))[None,:]


    from matplotlib import pyplot

    from string import ascii_lowercase
    from random import choice

    letters = [choice(ascii_lowercase) for _ in range(N)]

    df1 = pd.DataFrame(A, columns=letters)
    df2 = pd.DataFrame(B, columns=letters)
