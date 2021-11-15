import copy
from .results import ImprovedTimeIterationResult, TimeIterationResult, AlgoResult
from dolo.misc.itprinter import IterationsPrinter
from dolo.numeric.optimize.newton import newton
import numpy as np


from .time_iteration_helpers import *


def time_iteration(
    model: Model,
    *,
    dr0: DecisionRule = None,  #
    discretization: dict = dict(),
    interpolation="cubic",
    verbose: bool = True,  #
    details: bool = True,  #
    ignore_constraints=False,  #
    trace: bool = False,  #
    maxit=1000,
    tol_η=1e-6,
    tol_ε=1e-6,
    inner_maxit=5,
    hook=None,
) -> TimeIterationResult:

    F = Euler(
        model, discretization=discretization, interpolation=interpolation, dr0=dr0
    )

    x0 = F.x0

    complementarities = not ignore_constraints

    trace_details = []

    verbit = verbose == "full"

    itprint = IterationsPrinter(
        ("n", int),
        ("f_x", (float, "εₙ=|f(xₙ)|")),
        ("d_x", (float, "ηₙ=|xₙ-xₙ₋₁|")),
        ("λ", (float, "λₙ=ηₙ/ηₙ₋₁")),
        ("Time", float),
        ("nit", int),
        verbose=verbose,
    )

    itprint.print_header("Time Iterations.")

    err_η_0 = numpy.nan

    n_x = len(model.symbols["controls"])

    for it in range(maxit):

        if hook:
            hook()

        t_start = time.time()

        if trace:
            trace_details.append({"dr": copy.deepcopy(F.dr)})

        r = F(x0, x0)
        err_ε = r.norm()

        # r,J = F.d_A(su)

        verbose = False

        R, J = F.d_A(x0)
        R, L = F.d_B(x0)

        x1, nit = newton(F.d_A, x0, maxit=inner_maxit)

        # baby-steps version
        # r, J = ( F.d_A(x0) )
        # dx = J.solve(r)
        # err_η = dx.norm()
        # x1 = x0 - dx

        dx = x1 - x0
        err_η = dx.norm()

        λ = err_η / err_η_0
        err_η_0 = err_η

        t_finish = time.time()
        elapsed = t_finish - t_start

        itprint.print_iteration(n=it, f_x=err_ε, d_x=err_η, λ=λ, Time=elapsed, nit=nit),

        if err_ε < tol_ε or err_η < tol_η:
            break

        x0 = x1

    dr = F.dr

    itprint.print_finished()

    if not details:
        return dr

    return TimeIterationResult(
        dr,
        it,
        complementarities,
        F.dprocess,
        err_η < tol_η,  # x_converged: bool
        tol_η,  # x_tol
        err_η,  #: float
        None,  # log: object # TimeIterationLog
        trace_details,  # trace: object #{Nothing,IterationTrace}
    )


def improved_time_iteration(
    model: Model,
    *,
    dr0: DecisionRule = None,  #
    discretization: dict = dict(),
    interpolation="cubic",
    verbose: bool = True,  #
    details: bool = True,  #
    ignore_constraints=False,  #
    maxbsteps=10,
    tol_ε=1e-8,
    tol_ν=1e-10,
    smaxit=500,
    maxit=1000,
    compute_radius=False,
    # invmethod="iti",
) -> ImprovedTimeIterationResult:

    F = Euler(
        model,
        discretization=discretization,
        interpolation=interpolation,
        dr0=dr0,
        ignore_constraints=ignore_constraints,
    )

    x0 = F.x0

    steps = 0.5 ** numpy.arange(maxbsteps)

    complementarities = not ignore_constraints

    itprint = IterationsPrinter(
        ("n", int),
        # ("εₙ=|f(xₙ)|", float),
        # ("ηₙ=|f(xₙ)-f(xₙ₋₁)|", float),
        ("f_x", (float, "εₙ=|f(xₙ)|")),
        ("d_x", (float, "ηₙ=|xₙ-xₙ₋₁|")),
        # ("Time_residuals", float),
        # ("Time_inversion", float),
        ("λ", (float, "λ≈|T'(xₙ)|")),
        ("Time", float),
        ("N_invert", int),
        verbose=verbose,
    )
    itprint.print_header("Improved Time Iterations.")

    for it in range(maxit):

        t_start = time.time()

        dr = F(x0, x0)
        err_ε = dr.norm()

        dr, J = F.d_A(x0)
        dr, L = F.d_B(x0)

        dr = J.solve(dr)
        L.ldiv(J)
        L *= -1

        # compute dx such that: (I-L).dx = dr
        dx = dr
        du = dr
        err_ν_0 = 1.0
        for n in range(smaxit):
            du = L @ du
            dx = dx + du
            err_ν = du.norm()
            λ = err_ν / err_ν_0
            err_ν_0 = err_ν
            if err_ν < tol_ν:
                break

        err_η = dx.norm()

        for i_bckstps, lam in enumerate(steps):
            x1 = x0 - dx * lam
            err = F(x1, x1).norm()
            if err < err_ε:
                break

        x0 = x1

        t_finish = time.time()

        itprint.print_iteration(
            n=it, f_x=err_ε, d_x=err_η, λ=λ, Time=t_finish - t_start, N_invert=n
        )
        if err_ε < tol_ε:
            # print(f"Error {k} = {err_ε} |")
            break

    # F.dr.set_values(x0)

    dr = F.dr

    itprint.print_finished()

    if not details:
        return dr
    else:
        if compute_radius:
            raise Exception("Not implemented.")
        else:
            lam = np.nan
        return ImprovedTimeIterationResult(
            dr, it, err_ε, err_η, err_ε < tol_ε, complementarities, lam, None, L
        )