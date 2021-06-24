Finite iteration
================

By default, dolo looks for solutions, where the time horizon of the optimizing agent is infinite. But it is possible to use it to solve finite horizon problems (only with time-iteration, so far). It requires to do two things:

- Construct a final decision rule to be used for the last period. This can be done with the `CustomDR` function. ResultingsIt is passed as an initial guess to `time_iteration`.
- Record the decision rules, obtained in each iteration.

!!! example

    ```
    from dolo import yaml_import, time_iteration, tabulate
    from dolo.numeric.decision_rule import CustomDR

    model = yaml_import("examples/models/consumption_savings.yaml")

    last_dr = CustomDR(
        {"c": "w"},
        model=model
    )

    T = 10

    result = time_iteration(model, 
        dr0=last_dr, 
        maxit=T,
        trace=True # keeps decision rules, from all iterations
    )

    # example to plot all decision rules
    from matplotlib import pyplot as plt
    for i,tr in enumerate(result.trace):
        dr = tr['dr']
        tab = tabulate(model, dr, 'w')
        plt.plot(tab['w'], tab['c'], label=f"t={T-i}")
    plt.legend(loc='upper left')
    plt.show()
    ```

    In the example above, {"c": "w"} stands for a functional identity, i.e. c(y,w) = w. It is a completely different meaning from `c: w` in the calibration section which means that the steady-state value of `c` is initialized to the steady-state value of `w`.


!!!alert

    The notation in CustomDR is not yet consistent, with the new timing conventions. In the example above it should be `c[t] = w[t]`. A commission, will be created to examine the creation of an issue, meant to coordinate the implementation of a solution.