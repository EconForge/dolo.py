
Sudden Stop Model
=================

In this notebook we replicate the baseline model exposed in

``From Sudden Stops to Fisherian Deflation, Quantitative Theory and Policy``
by **Anton Korinek and Enrique G. Mendoza**

The file ``sudden_stop.yaml`` which is printed below, describes the
model, and must be included in the same directory as this notebook.

importing necessary functions
-----------------------------

.. code:: python

    %pylab inline

.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    from dolo import *
    from dolo.algos.dtmscc.time_iteration import time_iteration
    from dolo.algos.dtmscc.simulations import plot_decision_rule, simulate
writing the model
-----------------

.. code:: python

    cd ../models

.. parsed-literal::

    C:\Users\Pablo\Documents\GitHub\dolo\examples\models


.. code:: python

    filename = 'https://raw.githubusercontent.com/EconForge/dolo/master/examples/models/compat/sudden_stop.yaml'
    filename = 'sudden_stop.yaml'
    # the model file is coded in a separate file called sudden_stop.yaml
    # note how the borrowing constraint is implemented as complementarity condition
    pcat(filename)



.. raw:: html

    <style>
        .source .hll { background-color: #ffffcc }
    .source  { background: #f8f8f8; }
    .source .c { color: #408080; font-style: italic } /* Comment */
    .source .err { border: 1px solid #FF0000 } /* Error */
    .source .k { color: #008000; font-weight: bold } /* Keyword */
    .source .o { color: #666666 } /* Operator */
    .source .cm { color: #408080; font-style: italic } /* Comment.Multiline */
    .source .cp { color: #BC7A00 } /* Comment.Preproc */
    .source .c1 { color: #408080; font-style: italic } /* Comment.Single */
    .source .cs { color: #408080; font-style: italic } /* Comment.Special */
    .source .gd { color: #A00000 } /* Generic.Deleted */
    .source .ge { font-style: italic } /* Generic.Emph */
    .source .gr { color: #FF0000 } /* Generic.Error */
    .source .gh { color: #000080; font-weight: bold } /* Generic.Heading */
    .source .gi { color: #00A000 } /* Generic.Inserted */
    .source .go { color: #888888 } /* Generic.Output */
    .source .gp { color: #000080; font-weight: bold } /* Generic.Prompt */
    .source .gs { font-weight: bold } /* Generic.Strong */
    .source .gu { color: #800080; font-weight: bold } /* Generic.Subheading */
    .source .gt { color: #0044DD } /* Generic.Traceback */
    .source .kc { color: #008000; font-weight: bold } /* Keyword.Constant */
    .source .kd { color: #008000; font-weight: bold } /* Keyword.Declaration */
    .source .kn { color: #008000; font-weight: bold } /* Keyword.Namespace */
    .source .kp { color: #008000 } /* Keyword.Pseudo */
    .source .kr { color: #008000; font-weight: bold } /* Keyword.Reserved */
    .source .kt { color: #B00040 } /* Keyword.Type */
    .source .m { color: #666666 } /* Literal.Number */
    .source .s { color: #BA2121 } /* Literal.String */
    .source .na { color: #7D9029 } /* Name.Attribute */
    .source .nb { color: #008000 } /* Name.Builtin */
    .source .nc { color: #0000FF; font-weight: bold } /* Name.Class */
    .source .no { color: #880000 } /* Name.Constant */
    .source .nd { color: #AA22FF } /* Name.Decorator */
    .source .ni { color: #999999; font-weight: bold } /* Name.Entity */
    .source .ne { color: #D2413A; font-weight: bold } /* Name.Exception */
    .source .nf { color: #0000FF } /* Name.Function */
    .source .nl { color: #A0A000 } /* Name.Label */
    .source .nn { color: #0000FF; font-weight: bold } /* Name.Namespace */
    .source .nt { color: #008000; font-weight: bold } /* Name.Tag */
    .source .nv { color: #19177C } /* Name.Variable */
    .source .ow { color: #AA22FF; font-weight: bold } /* Operator.Word */
    .source .w { color: #bbbbbb } /* Text.Whitespace */
    .source .mf { color: #666666 } /* Literal.Number.Float */
    .source .mh { color: #666666 } /* Literal.Number.Hex */
    .source .mi { color: #666666 } /* Literal.Number.Integer */
    .source .mo { color: #666666 } /* Literal.Number.Oct */
    .source .sb { color: #BA2121 } /* Literal.String.Backtick */
    .source .sc { color: #BA2121 } /* Literal.String.Char */
    .source .sd { color: #BA2121; font-style: italic } /* Literal.String.Doc */
    .source .s2 { color: #BA2121 } /* Literal.String.Double */
    .source .se { color: #BB6622; font-weight: bold } /* Literal.String.Escape */
    .source .sh { color: #BA2121 } /* Literal.String.Heredoc */
    .source .si { color: #BB6688; font-weight: bold } /* Literal.String.Interpol */
    .source .sx { color: #008000 } /* Literal.String.Other */
    .source .sr { color: #BB6688 } /* Literal.String.Regex */
    .source .s1 { color: #BA2121 } /* Literal.String.Single */
    .source .ss { color: #19177C } /* Literal.String.Symbol */
    .source .bp { color: #008000 } /* Name.Builtin.Pseudo */
    .source .vc { color: #19177C } /* Name.Variable.Class */
    .source .vg { color: #19177C } /* Name.Variable.Global */
    .source .vi { color: #19177C } /* Name.Variable.Instance */
    .source .il { color: #666666 } /* Literal.Number.Integer.Long */
        </style>
        <table class="sourcetable"><tr><td class="linenos"><div class="linenodiv"><pre> 1
     2
     3
     4
     5
     6
     7
     8
     9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    31
    32
    33
    34
    35
    36
    37
    38
    39
    40
    41
    42
    43
    44
    45
    46
    47
    48
    49
    50
    51
    52
    53
    54
    55
    56
    57
    58
    59
    60
    61
    62
    63
    64
    65
    66
    67
    68
    69
    70
    71
    72
    73
    74
    75
    76</pre></div></td><td class="code"><div class="source"><pre><span class="c1"># This file adapts the model described in</span>
    <span class="c1"># &quot;From Sudden Stops to Fisherian Deflation, Quantitative Theory and Policy&quot;</span>
    <span class="c1"># by Anton Korinek and Enrique G. Mendoza</span>

    <span class="l-Scalar-Plain">name</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">Sudden Stop (General)</span>

    <span class="l-Scalar-Plain">model_spec</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">mfga</span>

    <span class="l-Scalar-Plain">symbols</span><span class="p-Indicator">:</span>

        <span class="l-Scalar-Plain">markov_states</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">y</span><span class="p-Indicator">]</span>
        <span class="l-Scalar-Plain">states</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">l</span><span class="p-Indicator">]</span>
        <span class="l-Scalar-Plain">controls</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">b</span><span class="p-Indicator">,</span> <span class="nv">lam</span><span class="p-Indicator">]</span>
        <span class="l-Scalar-Plain">auxiliaries</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">c</span><span class="p-Indicator">]</span>
        <span class="l-Scalar-Plain">values</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">V</span><span class="p-Indicator">,</span> <span class="nv">Vc</span><span class="p-Indicator">]</span>
        <span class="l-Scalar-Plain">parameters</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">beta</span><span class="p-Indicator">,</span> <span class="nv">R</span><span class="p-Indicator">,</span> <span class="nv">sigma</span><span class="p-Indicator">,</span> <span class="nv">a</span><span class="p-Indicator">,</span> <span class="nv">mu</span><span class="p-Indicator">,</span> <span class="nv">kappa</span><span class="p-Indicator">,</span> <span class="nv">delta_y</span><span class="p-Indicator">,</span> <span class="nv">pi</span><span class="p-Indicator">,</span> <span class="nv">lam_inf</span><span class="p-Indicator">]</span>

    <span class="l-Scalar-Plain">equations</span><span class="p-Indicator">:</span>

        <span class="l-Scalar-Plain">transition</span><span class="p-Indicator">:</span>

            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">l = b(-1)</span>

        <span class="l-Scalar-Plain">arbitrage</span><span class="p-Indicator">:</span>

            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">lam = b/c</span>
            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">beta*(c(1)/c)^(-sigma)*R - 1    |  lam_inf &lt;= lam &lt;= inf</span>


        <span class="l-Scalar-Plain">auxiliary</span><span class="p-Indicator">:</span>

            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">c = 1 + y + l*R - b</span>

        <span class="l-Scalar-Plain">value</span><span class="p-Indicator">:</span>

            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">V = c^(1.0-sigma)/(1.0-sigma) + beta*V(1)</span>
            <span class="p-Indicator">-</span> <span class="l-Scalar-Plain">Vc = c^(1.0-sigma)/(1.0-sigma)</span>


    <span class="l-Scalar-Plain">discrete_transition</span><span class="p-Indicator">:</span>

        <span class="l-Scalar-Plain">MarkovChain</span><span class="p-Indicator">:</span>

            <span class="p-Indicator">-</span> <span class="p-Indicator">[[</span> <span class="nv">1.0-delta_y</span> <span class="p-Indicator">],</span>  <span class="c1"># bad state</span>
               <span class="p-Indicator">[</span> <span class="nv">1.0</span> <span class="p-Indicator">]]</span>          <span class="c1"># good state</span>

            <span class="p-Indicator">-</span> <span class="p-Indicator">[[</span> <span class="nv">0.5</span><span class="p-Indicator">,</span> <span class="nv">1-0.5</span> <span class="p-Indicator">],</span>   <span class="c1"># probabilities   [p(L|L), p(H|L)]</span>
               <span class="p-Indicator">[</span> <span class="nv">0.5</span><span class="p-Indicator">,</span> <span class="nv">0.5</span> <span class="p-Indicator">]]</span>     <span class="c1"># probabilities   [p(L|H), p(H|H)]</span>

    <span class="l-Scalar-Plain">calibration</span><span class="p-Indicator">:</span>

        <span class="l-Scalar-Plain">beta</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.95</span>
        <span class="l-Scalar-Plain">R</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">1.03</span>
        <span class="l-Scalar-Plain">sigma</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">2.0</span>
        <span class="l-Scalar-Plain">a</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">1/3</span>
        <span class="l-Scalar-Plain">mu</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.8</span>
        <span class="l-Scalar-Plain">kappa</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">1.3</span>
        <span class="l-Scalar-Plain">delta_y</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.03</span>
        <span class="l-Scalar-Plain">pi</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.05</span>
        <span class="l-Scalar-Plain">lam_inf</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">-0.2</span>
        <span class="l-Scalar-Plain">y</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">1.0</span>
        <span class="l-Scalar-Plain">c</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">1.0 + y</span>
        <span class="l-Scalar-Plain">b</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.0</span>
        <span class="l-Scalar-Plain">l</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.0</span>
        <span class="l-Scalar-Plain">lam</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">0.0</span>

        <span class="l-Scalar-Plain">V</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">c^(1.0-sigma)/(1.0-sigma)/(1.0-beta)</span>
        <span class="l-Scalar-Plain">Vc</span><span class="p-Indicator">:</span> <span class="l-Scalar-Plain">c^(1.0-sigma)/(1.0-sigma)</span>

    <span class="l-Scalar-Plain">options</span><span class="p-Indicator">:</span>

        <span class="l-Scalar-Plain">approximation_space</span><span class="p-Indicator">:</span>

            <span class="l-Scalar-Plain">a</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">-1.0</span><span class="p-Indicator">]</span>
            <span class="l-Scalar-Plain">b</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span> <span class="nv">1.0</span><span class="p-Indicator">]</span>
            <span class="l-Scalar-Plain">orders</span><span class="p-Indicator">:</span> <span class="p-Indicator">[</span><span class="nv">10</span><span class="p-Indicator">]</span>
    </pre></div>
    </td></tr></table>




importing the model
-------------------

Note, that residuals, are not zero at the calibration we supply. This is
because the representative agent is impatient and we have
:math:`\beta<1/R`. In this case it doesn't matter.

By default, the calibrated value for endogenous variables are used as a
(constant) starting point for the decision rules.

.. code:: python

    model = yaml_import('sudden_stop.yaml')
    model

.. parsed-literal::

    Model type detected as 'dtmscc'




.. parsed-literal::


    Model object:
    ------------

    - name: "Sudden Stop (General)"
    - type: "dtmscc"
    - file: "sudden_stop.yaml

    - residuals:

        transition
            1   : 0.0000 : l = b(-1)

        arbitrage
            1   : 0.0000 : lam = b/c
            2   : [31m-0.0215[0m : beta*(c(1)/c)**(-sigma)*R - 1    |  lam_inf <= lam <= inf

        auxiliary
            1   : 0.0000 : c = 1 + y + l*R - b

        value
            1   : 0.0000 : V = c**(1.0-sigma)/(1.0-sigma) + beta*V(1)
            2   : 0.0000 : Vc = c**(1.0-sigma)/(1.0-sigma)




.. code:: python

    # to avoid numerical glitches we choose a relatively high number of grid points
    mdr = time_iteration(model, verbose=True, orders=[1000])

.. parsed-literal::

    Solving WITH complementarities.
    ------------------------------------------------
    | N   |  Error     | Gain     | Time     | nit |
    ------------------------------------------------
    |   1 |  5.014e-01 |      nan |    1.878 |   7 |
    |   2 |  1.600e-01 |    0.319 |    0.235 |   6 |
    |   3 |  7.472e-02 |    0.467 |    0.221 |   6 |
    |   4 |  4.065e-02 |    0.544 |    0.198 |   5 |
    |   5 |  2.388e-02 |    0.587 |    0.204 |   5 |
    |   6 |  1.933e-02 |    0.809 |    0.354 |   9 |
    |   7 |  1.609e-02 |    0.832 |    0.234 |   6 |
    |   8 |  1.370e-02 |    0.852 |    0.200 |   5 |
    |   9 |  1.187e-02 |    0.867 |    0.148 |   4 |
    |  10 |  1.049e-02 |    0.883 |    0.112 |   3 |
    |  11 |  9.381e-03 |    0.894 |    0.138 |   3 |
    |  12 |  8.467e-03 |    0.903 |    0.120 |   3 |
    |  13 |  7.711e-03 |    0.911 |    0.126 |   3 |
    |  14 |  7.060e-03 |    0.916 |    0.123 |   3 |
    |  15 |  6.503e-03 |    0.921 |    0.078 |   2 |
    |  16 |  6.016e-03 |    0.925 |    0.102 |   2 |
    |  17 |  4.611e-03 |    0.766 |    0.083 |   2 |
    |  18 |  8.356e-04 |    0.181 |    0.101 |   2 |
    |  19 |  8.879e-05 |    0.106 |    0.056 |   1 |
    |  20 |  1.449e-05 |    0.163 |    0.060 |   1 |
    |  21 |  2.483e-06 |    0.171 |    0.056 |   1 |
    |  22 |  2.605e-07 |    0.105 |    0.056 |   1 |
    ------------------------------------------------
    Elapsed: 4.91300010681 seconds.
    ------------------------------------------------


.. code:: python

    # produce the plots
    n_steps = 100

    figure(figsize(10,6))
    subplot(121)
    plot_decision_rule(model, mdr, 'l', 'b', i0=0, n_steps=n_steps, label='$b_t$ (bad state)' )
    plot_decision_rule(model, mdr, 'l', 'b', i0=1, n_steps=n_steps, label='$b_t$ (good state)' )
    plot_decision_rule(model, mdr, 'l', 'l', i0=1, n_steps=n_steps, linestyle='--', color='black' )
    #plot(df['l'], df['l'], linestyle='--', color='black')

    # to plot the borrowing limit, we produce a dataframe df which contains all series
    # (note that we don't supply a variable name to plot, only the state 'l')

    lam_inf = model.get_calibration('lam_inf')
    df = plot_decision_rule(model, mdr, 'l', i0=0, n_steps=n_steps)
    plot(df['l'], lam_inf*df['c'], linestyle='--', color='black')

    xlabel('$l_t$')

    legend(loc= 'upper left')


    subplot(122)
    plot_decision_rule(model, mdr, 'l', 'c', i0=0, n_steps=n_steps, label='$c_t$ (bad state)' )
    plot_decision_rule(model, mdr, 'l', 'c', i0=1, n_steps=n_steps, label='$c_t$ (good state)' )
    legend(loc= 'lower right')
    xlabel('$l_t$')

    suptitle("Decision Rules")




.. parsed-literal::

    <matplotlib.text.Text at 0x179751d0>




.. image:: sudden_stop_files%5Csudden_stop_10_1.png


.. code:: python

    ## stochastic simulations
.. code:: python

    i_0 = 1 # we start from the good state
    sim = simulate(model, mdr, i_0, s0=0.5, n_exp=1, horizon=100) # markov_indices=markov_indices)
.. code:: python

    subplot(211)
    plot(sim['y'])
    subplot(212)
    plot(sim['b'])



.. parsed-literal::

    [<matplotlib.lines.Line2D at 0x18f07668>]




.. image:: sudden_stop_files%5Csudden_stop_13_1.png


Sensitivity analysis
--------------------

Here we want to compare the saving behaviour as a function of risk
aversion :math:`\sigma`. We contrast the baseline :math:`\sigma=2` with
the high aversion scenario :math:`\sigma=16`.

.. code:: python

    # we solve the model with sigma=16
    model.set_calibration(sigma=16.0)
    mdr_high_gamma = time_iteration(model, verbose=True, orders=[1000])

.. parsed-literal::

    Solving WITH complementarities.
    ------------------------------------------------
    | N   |  Error     | Gain     | Time     | nit |
    ------------------------------------------------
    |   1 |  5.133e-01 |      nan |    0.395 |  10 |
    |   2 |  1.703e-01 |    0.332 |    0.295 |   8 |
    |   3 |  8.435e-02 |    0.495 |    0.284 |   7 |
    |   4 |  5.005e-02 |    0.593 |    0.277 |   7 |
    |   5 |  3.292e-02 |    0.658 |    0.281 |   7 |
    |   6 |  2.313e-02 |    0.703 |    0.281 |   7 |
    |   7 |  1.702e-02 |    0.736 |    0.268 |   7 |
    |   8 |  1.295e-02 |    0.761 |    0.267 |   7 |
    |   9 |  1.011e-02 |    0.780 |    0.286 |   7 |
    |  10 |  8.045e-03 |    0.796 |    0.271 |   7 |
    |  11 |  6.501e-03 |    0.808 |    0.283 |   7 |
    |  12 |  5.316e-03 |    0.818 |    0.268 |   7 |
    |  13 |  4.387e-03 |    0.825 |    0.249 |   6 |
    |  14 |  3.647e-03 |    0.831 |    0.294 |   7 |
    |  15 |  3.048e-03 |    0.836 |    0.279 |   7 |
    |  16 |  2.558e-03 |    0.839 |    0.256 |   6 |
    |  17 |  2.206e-03 |    0.863 |    0.235 |   6 |
    |  18 |  2.010e-03 |    0.911 |    0.334 |   6 |
    |  19 |  1.842e-03 |    0.916 |    0.330 |   5 |
    |  20 |  1.699e-03 |    0.922 |    0.307 |   5 |
    |  21 |  1.580e-03 |    0.930 |    0.314 |   5 |
    |  22 |  1.472e-03 |    0.932 |    0.316 |   5 |
    |  23 |  1.374e-03 |    0.933 |    0.302 |   5 |
    |  24 |  1.289e-03 |    0.938 |    0.303 |   5 |
    |  25 |  1.210e-03 |    0.939 |    0.316 |   5 |
    |  26 |  1.137e-03 |    0.940 |    0.310 |   5 |
    |  27 |  1.073e-03 |    0.944 |    0.263 |   4 |
    |  28 |  1.013e-03 |    0.944 |    0.259 |   4 |
    |  29 |  9.575e-04 |    0.945 |    0.202 |   3 |
    |  30 |  9.075e-04 |    0.948 |    0.204 |   3 |
    |  31 |  8.600e-04 |    0.948 |    0.194 |   3 |
    |  32 |  8.166e-04 |    0.950 |    0.211 |   3 |
    |  33 |  7.764e-04 |    0.951 |    0.185 |   3 |
    |  34 |  7.384e-04 |    0.951 |    0.186 |   3 |
    |  35 |  7.035e-04 |    0.953 |    0.204 |   3 |
    |  36 |  6.705e-04 |    0.953 |    0.145 |   2 |
    |  37 |  6.396e-04 |    0.954 |    0.150 |   2 |
    |  38 |  6.108e-04 |    0.955 |    0.152 |   2 |
    |  39 |  5.835e-04 |    0.955 |    0.142 |   2 |
    |  40 |  5.579e-04 |    0.956 |    0.138 |   2 |
    |  41 |  5.338e-04 |    0.957 |    0.153 |   2 |
    |  42 |  5.110e-04 |    0.957 |    0.134 |   2 |
    |  43 |  4.895e-04 |    0.958 |    0.151 |   2 |
    |  44 |  4.691e-04 |    0.958 |    0.135 |   2 |
    |  45 |  4.499e-04 |    0.959 |    0.149 |   2 |
    |  46 |  4.316e-04 |    0.959 |    0.135 |   2 |
    |  47 |  4.143e-04 |    0.960 |    0.138 |   2 |
    |  48 |  3.978e-04 |    0.960 |    0.143 |   2 |
    |  49 |  3.821e-04 |    0.961 |    0.152 |   2 |
    |  50 |  3.598e-04 |    0.941 |    0.133 |   2 |
    |  51 |  3.132e-04 |    0.871 |    0.151 |   2 |
    |  52 |  2.476e-04 |    0.790 |    0.146 |   2 |
    |  53 |  1.782e-04 |    0.720 |    0.134 |   2 |
    |  54 |  1.190e-04 |    0.668 |    0.141 |   2 |
    |  55 |  7.541e-05 |    0.634 |    0.140 |   2 |
    |  56 |  4.634e-05 |    0.615 |    0.176 |   2 |
    |  57 |  2.802e-05 |    0.605 |    0.145 |   2 |
    |  58 |  1.684e-05 |    0.601 |    0.146 |   2 |
    |  59 |  1.010e-05 |    0.600 |    0.086 |   1 |
    |  60 |  6.072e-06 |    0.601 |    0.081 |   1 |
    |  61 |  3.659e-06 |    0.603 |    0.077 |   1 |
    |  62 |  2.211e-06 |    0.604 |    0.098 |   1 |
    |  63 |  1.340e-06 |    0.606 |    0.081 |   1 |
    |  64 |  8.141e-07 |    0.607 |    0.086 |   1 |
    ------------------------------------------------
    Elapsed: 13.4159998894 seconds.
    ------------------------------------------------


.. parsed-literal::

    [33mUserWarning[0m:c:\users\pablo\documents\github\dolo\dolo\numeric\optimize\newton.py:150
        Did not converge


.. code:: python

    # now we compare the decision rules with low and high risk aversion
    plot_decision_rule(model, mdr, 'l', 'b', i0=0, n_steps=n_steps, label='$b_t$ (bad)' )
    plot_decision_rule(model, mdr, 'l', 'b', i0=1, n_steps=n_steps, label='$b_t$ (good)' )
    plot_decision_rule(model, mdr_high_gamma, 'l', 'b', i0=0, n_steps=n_steps, label='$b_t$ (bad) [high gamma]' )
    plot_decision_rule(model, mdr_high_gamma, 'l', 'b', i0=1, n_steps=n_steps, label='$b_t$ (good) [high gamma]' )
    plot(df['l'], df['l'], linestyle='--', color='black')
    plot(df['l'], -0.2*df['c'], linestyle='--', color='black')
    legend(loc= 'upper left')



.. parsed-literal::

    <matplotlib.legend.Legend at 0x192abac8>




.. image:: sudden_stop_files%5Csudden_stop_16_1.png
