import numpy as np

# doesn't seem to support keyword arguments yet
# from multipledispatch import dispatch
# multimethod = dispatch()
from dolo.misc.multimethod import multimethod
from numba import generated_jit
import xarray

from dolo.compiler.model import Model
from dolo.numeric.grids import UniformCartesianGrid
from dolo.numeric.decision_rule import DecisionRule
from dolo.numeric.processes import MarkovChain, IIDProcess, DiscretizedIIDProcess
from dolo.numeric.grids import CartesianGrid, UnstructuredGrid, EmptyGrid


@generated_jit(nopython=True)
def trembling_hand(A: "N*n1*...*nd", x: "N*d", w: "float"):

    # we could generate that automatically for orders >=3
    if A.ndim == 2:

        def fun(A: "N* *...", x: "N*d", w: "float"):

            N, n0 = A.shape
            δ0 = 1 / (n0 - 1)

            for n in range(N):

                x0 = x[n, 0]
                x0 = min(max(x0, 0), 1)
                q0 = np.floor_divide(x0, δ0)
                q0 = max(0, q0)
                q0 = min(q0, n0 - 2)

                λ0 = x0 / δ0 - q0  # ∈[0,1[ by construction

                q0_ = int(q0)

                A[n, q0_] += (1 - λ0) * w
                A[n, q0_ + 1] += λ0 * w

        return fun

    elif A.ndim == 3:

        def fun(A: "N* *...", x: "N*d", w: "float"):

            N, n0, n1 = A.shape
            δ0 = 1 / (n0 - 1)
            δ1 = 1 / (n1 - 1)

            for n in range(N):

                x0 = x[n, 0]
                x0 = min(max(x0, 0), 1)
                q0 = np.floor_divide(x0, δ0)
                q0 = max(0, q0)
                q0 = min(q0, n0 - 2)

                x1 = x[n, 1]
                x1 = min(max(x1, 0), 1)
                q1 = np.floor_divide(x1, δ1)
                q1 = max(0, q1)
                q1 = min(q1, n1 - 2)

                λ0 = (x0 - q0 * δ0) / δ0  # ∈[0,1[ by construction
                q0_ = int(q0)

                λ1 = (x1 - q1 * δ1) / δ1  # ∈[0,1[ by construction
                q1_ = int(q1)

                A[n, q0_, q1_] += (1 - λ0) * (1 - λ1) * w
                A[n, q0_, q1_ + 1] += (1 - λ0) * (λ1) * w
                A[n, q0_ + 1, q1_] += (λ0) * (1 - λ1) * w
                A[n, q0_ + 1, q1_ + 1] += (λ0) * (λ1) * w

        return fun

    else:
        raise Exception("Unsupported.")


# TODO: add default options for endo_grid, exo_grid, dp
@multimethod
def ergodic_distribution(model: Model, dr: DecisionRule):
    return ergodic_distribution(model, dr, dr.exo_grid, dr.endo_grid, dr.dprocess)


@multimethod
def ergodic_distribution(
    model: Model,
    dr: DecisionRule,
    exo_grid: UnstructuredGrid,
    endo_grid: UniformCartesianGrid,
    dp: MarkovChain,
    compute_μ=True,
):

    N_m = exo_grid.n_nodes
    N_s = endo_grid.n_nodes
    dims_s = tuple(endo_grid.n)  # we have N_s = prod(dims_s)
    s = endo_grid.nodes

    N = N_m * N_s
    Π = np.zeros((N_m, N_m, N_s) + dims_s)

    g = model.functions["transition"]
    p = model.calibration["parameters"]

    a = endo_grid.min
    b = endo_grid.max

    for i_m in range(N_m):
        m = exo_grid.node(i_m)
        x: np.array = dr(i_m, s)
        for i_M in range(dp.n_inodes(i_m)):
            w = dp.iweight(i_m, i_M)
            M = dp.inode(i_m, i_M)
            S = g(m, s, x, M, p)

            # renormalize
            S = (S - a[None, :]) / (b[None, :] - a[None, :])

            # allocate weights
            dest = Π[i_m, i_M, ...]
            trembling_hand(dest, S, w)

    # these seem to be a bit expensive...
    Π = Π.reshape((N_m, N_m, N_s, N_s))
    Π = Π.swapaxes(1, 2)
    Π = Π.reshape((N, N))

    if not compute_μ:
        return Π.reshape((N_m, N_s, N_m, N_s))
    else:
        B = np.zeros(N)
        B[-1] = 1.0
        A = Π.T - np.eye(Π.shape[0])
        A[-1, :] = 1.0
        μ = np.linalg.solve(A, B)

        μ = μ.reshape((N_m,) + dims_s)
        labels = [
            np.linspace(endo_grid.min[i], endo_grid.max[i], endo_grid.n[i])
            for i in range(len(endo_grid.max))
        ]
        μ = xarray.DataArray(
            μ,
            [("i_m", np.arange(N_m))]
            + list(
                {s: labels[i] for i, s in enumerate(model.symbols["states"])}.items()
            ),
        )
        return Π.reshape((N_m, N_s, N_m, N_s)), μ


@multimethod
def ergodic_distribution(
    model: Model,
    dr: DecisionRule,
    exo_grid: EmptyGrid,
    endo_grid: CartesianGrid,
    dp: DiscretizedIIDProcess,
    compute_μ=True,
):

    N_s = endo_grid.n_nodes
    dims_s = tuple(endo_grid.n)  # we have N_s = prod(dims_s)
    s = endo_grid.nodes

    m = model.calibration["exogenous"]
    Π = np.zeros((N_s,) + dims_s)

    g = model.functions["transition"]
    p = model.calibration["parameters"]

    a = endo_grid.min
    b = endo_grid.max

    x: np.array = dr(s)

    for i_M in range(dp.n_inodes(0)):

        w = dp.iweight(0, i_M)
        M = dp.inode(0, i_M)
        S = g(m, s, x, M, p)
        # renormalize
        S = (S - a[None, :]) / (b[None, :] - a[None, :])
        # allocate weights
        trembling_hand(Π, S, w)

    Π = Π.reshape((N_s, N_s))

    if not compute_μ:
        return Π

    else:
        B = np.zeros(N_s)
        B[-1] = 1.0
        A = Π.T - np.eye(Π.shape[0])
        A[-1, :] = 1.0
        μ = np.linalg.solve(A, B)
        μ = μ.reshape(dims_s)

        labels = [
            np.linspace(endo_grid.min[i], endo_grid.max[i], endo_grid.n[i])
            for i in range(len(endo_grid.max))
        ]
        μ = xarray.DataArray(
            μ,
            list({s: labels[i] for i, s in enumerate(model.symbols["states"])}.items()),
        )

        return Π, μ
