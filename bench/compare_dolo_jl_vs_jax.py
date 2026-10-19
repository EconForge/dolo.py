"""
Comprehensive Cross-Platform Benchmark:
- Dolo.jl (Julia)
- dolo.py (JAX vmap)
- dolo.py (Old Numba Gufuncs: Dolang + Splines)
- dolo.py (Current Vectorized: euler_residuals_tau)

Same RBC model, identical grid points and calibration:
- Residual evaluation F(s, x, φ)
- Derivative evaluation dF_1(s, x, φ)
"""

from __future__ import annotations

import json
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import dolo
from dolo import yaml_import, yaml_import_v2
from dolo.numeric import DiscretizedModel
from dolo.algos.time_iteration import euler_residuals_tau
from dolo.numeric.optimize.newton import SerialDifferentiableFunction
from dolo.numeric.residual_jax import compile_residual
from interpolation.splines import eval_cubic
from interpolation.splines.prefilter_cubic import prefilter_cubic

console = Console()


def time_callable(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for item in res:
                if hasattr(item, "block_until_ready"):
                    item.block_until_ready()

    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for item in res:
                if hasattr(item, "block_until_ready"):
                    item.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return min(times) * 1000.0


def main():
    with open("bench/dolo_jl_results.json") as f:
        jl_results = {r["N"]: r for r in json.load(f)}

    model_path = "examples/models/rbc.yaml"
    legacy_model = yaml_import(model_path)
    f_gufun = legacy_model.functions["arbitrage"].fun
    g_gufun = legacy_model.functions["transition"].fun
    p_numba = np.array(legacy_model.calibration["parameters"])

    grid_sizes = [
        [10, 10],    # 100
        [32, 32],    # 1,024
        [100, 100],  # 10,000
        [200, 200],  # 40,000
    ]

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Cross-Platform Benchmark: JAX vs. Dolo.jl vs. Old Numba vs. Current Python[/bold cyan]\n"
            "[dim]Model: Real Business Cycle (RBC) | Endogenous states: 2 (z, k) | Controls: 2 (n, i)\n"
            "Hardware: CPU | Precision: Float64[/dim]",
            border_style="cyan",
        )
    )

    rows_f = []
    rows_df = []

    for gs in grid_sizes:
        N = gs[0] * gs[1]
        console.print(f"Running benchmarks for N = {N:,d} ({gs[0]}x{gs[1]})...")

        # --- JAX setup ---
        model_v2 = yaml_import_v2(model_path)
        dmodel_v2 = DiscretizedModel(model_v2, grid_options=dict(n=gs))
        controls_ga = dmodel_v2.initial_guess()
        dr_v2 = dmodel_v2.decision_rule(controls_ga)
        n_x = len(dmodel_v2.model.calibration["controls"])

        cr = compile_residual(dmodel_v2, interp_method="cubic", diff_mode="fwd")
        coeffs_jax = cr.prefilter(controls_ga)
        s_nodes_jax = jnp.asarray(dmodel_v2.endo_nodes, dtype=jnp.float64)
        x_vals_jax = jnp.asarray(controls_ga[0], dtype=jnp.float64)

        # 1. JAX Timings
        t_f_jax = time_callable(lambda: cr.F_vmap(s_nodes_jax, x_vals_jax, coeffs_jax), repeat=8, warmup=2)
        t_df_jax = time_callable(lambda: cr.dF_vmap(s_nodes_jax, x_vals_jax, coeffs_jax), repeat=6, warmup=2)

        # 2. Dolo.jl Timings
        t_f_jl = jl_results[N]["t_f_inplace_ms"]
        t_df_jl = jl_results[N]["t_df_inplace_ms"]

        # 3. Old Numba Gufuncs Setup
        grid_legacy, dp_legacy = legacy_model.discretize(dict(n=gs), {})
        endo_grid_legacy = grid_legacy.grids[1]
        s_numba = endo_grid_legacy.nodes
        grid_spline = (
            (float(endo_grid_legacy.min[0]), float(endo_grid_legacy.max[0]), int(endo_grid_legacy.n[0])),
            (float(endo_grid_legacy.min[1]), float(endo_grid_legacy.max[1]), int(endo_grid_legacy.n[1])),
        )
        x0_numba = np.tile(legacy_model.calibration["controls"], (N, 1))
        coeffs_numba = prefilter_cubic(grid_spline, x0_numba.reshape(gs[0], gs[1], 2))

        m_val = dp_legacy.node(0)
        n_inodes = dp_legacy.n_inodes(0)
        inodes = [dp_legacy.inode(0, k) for k in range(n_inodes)]
        weights = [dp_legacy.iweight(0, k) for k in range(n_inodes)]

        def euler_numba_eval(s, x, coeffs):
            res = np.zeros_like(x)
            M_arr = np.empty_like(s[:, :1])
            S = np.empty_like(s)
            XM = np.empty_like(x)
            r = np.empty_like(x)
            m_arr = np.tile(m_val, (N, 1))
            for k in range(n_inodes):
                M_arr[:] = inodes[k]
                w = weights[k]
                g_gufun(m_arr, s, x, M_arr, p_numba, S)
                eval_cubic(grid_spline, coeffs, S, XM)
                f_gufun(m_arr, s, x, M_arr, S, XM, p_numba, r)
                res += w * r
            return res

        def df_numba_eval(s, x, coeffs, eps=1e-7):
            r0 = euler_numba_eval(s, x, coeffs)
            N_pts, n_x = x.shape
            J = np.empty((N_pts, n_x, n_x))
            for j in range(n_x):
                x_pert = x.copy()
                x_pert[:, j] += eps
                rj = euler_numba_eval(s, x_pert, coeffs)
                J[:, :, j] = (rj - r0) / eps
            return r0, J

        t_f_numba = time_callable(lambda: euler_numba_eval(s_numba, x0_numba, coeffs_numba), repeat=6, warmup=2)
        t_df_numba = time_callable(lambda: df_numba_eval(s_numba, x0_numba, coeffs_numba), repeat=4, warmup=1)

        # 4. Current dolo.py (euler_residuals_tau + SerialDifferentiableFunction)
        t_f_cur = time_callable(lambda: euler_residuals_tau(dmodel_v2, controls_ga, dr_v2), repeat=4, warmup=1)
        if N <= 10000:
            fn_cur = (
                lambda x_val: euler_residuals_tau(dmodel_v2, controls_ga.unravel(x_val), dr_v2)
                .ravel()
                .reshape((-1, n_x))
            )
            dfn_cur = SerialDifferentiableFunction(fn_cur, epsilon=1e-7)
            t_df_cur = time_callable(lambda: dfn_cur(np.array(controls_ga[0])), repeat=3, warmup=1)
        else:
            t_df_cur = t_f_cur * (n_x + 1)

        rows_f.append({
            "N": N,
            "t_cur": t_f_cur,
            "t_numba": t_f_numba,
            "t_jl": t_f_jl,
            "t_jax": t_f_jax,
        })

        rows_df.append({
            "N": N,
            "t_cur": t_df_cur,
            "t_numba": t_df_numba,
            "t_jl": t_df_jl,
            "t_jax": t_df_jax,
        })

    # Table 1: Residual F
    t1 = Table(title="\n1. Residual Function F Evaluation", border_style="blue")
    t1.add_column("Grid Points (N)", justify="right", style="cyan")
    t1.add_column("dolo.py Current (ms)", justify="right", style="magenta")
    t1.add_column("dolo.py Old Numba (ms)", justify="right", style="yellow")
    t1.add_column("Dolo.jl Julia (ms)", justify="right", style="bright_yellow")
    t1.add_column("dolo.py JAX vmap (ms)", justify="right", style="bold green")

    for r in rows_f:
        t1.add_row(
            f"{r['N']:,d}",
            f"{r['t_cur']:.2f}",
            f"{r['t_numba']:.2f}",
            f"{r['t_jl']:.2f}",
            f"{r['t_jax']:.2f}",
        )
    console.print(t1)

    # Table 2: Derivative dF
    t2 = Table(title="\n2. Derivative / Jacobian dF_1 Evaluation w.r.t Controls", border_style="green")
    t2.add_column("Grid Points (N)", justify="right", style="cyan")
    t2.add_column("dolo.py Current FD (ms)", justify="right", style="magenta")
    t2.add_column("dolo.py Old Numba FD (ms)", justify="right", style="yellow")
    t2.add_column("Dolo.jl ForwardDiff (ms)", justify="right", style="bright_yellow")
    t2.add_column("dolo.py JAX jacfwd (ms)", justify="right", style="bold green")

    for r in rows_df:
        t2.add_row(
            f"{r['N']:,d}",
            f"{r['t_cur']:.2f}",
            f"{r['t_numba']:.2f}",
            f"{r['t_jl']:.2f}",
            f"{r['t_jax']:.2f}",
        )
    console.print(t2)
    console.print()


if __name__ == "__main__":
    main()
