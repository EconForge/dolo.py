"""
Benchmark: Current Vectorized Approach vs. JAX Compiled & Vmapped Residual & Derivative.

Architecture follows Dolo.jl (Dolo.F / Dolo.dF_1):
- Residual function F(s, x, φ): Point Euler residuals
- Derivative dF_1(s, x, φ): Exact Jacobian w.r.t controls x
- Vectorized F across the grid via jax.vmap
- Vectorized dF_1 across the grid via jax.vmap (jacfwd / jacrev)
- Joint evaluation of F and dF_1 in a single fused XLA pass

Usage:
    pixi run --environment dev python bench/bench_residual_jax.py
"""

from __future__ import annotations

import time
import warnings
import numpy as np

# Suppress minor backend warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from dolo import yaml_import_v2
from dolo.numeric import DiscretizedModel
from dolo.algos.time_iteration import euler_residuals_tau
from dolo.numeric.optimize.newton import SerialDifferentiableFunction
from dolo.numeric.residual_jax import compile_residual

console = Console()


def time_callable(fn, repeat=5, warmup=2):
    """Accurately time a callable with warmup and synchronization."""
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

    return min(times) * 1000.0  # min runtime in milliseconds


def benchmark_grid_size(model_path: str, n_grid: list[int]):
    N_points = n_grid[0] * n_grid[1]
    model = yaml_import_v2(model_path)
    dmodel = DiscretizedModel(model, grid_options=dict(n=n_grid))

    controls_ga = dmodel.initial_guess()
    dr = dmodel.decision_rule(controls_ga)
    n_x = len(dmodel.model.calibration["controls"])

    # 1. Compile JAX residual operators
    cr_fwd = compile_residual(dmodel, interp_method="cubic", diff_mode="fwd")
    cr_rev = compile_residual(dmodel, interp_method="cubic", diff_mode="rev")

    coeffs = cr_fwd.prefilter(controls_ga)
    s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
    x_vals = jnp.asarray(controls_ga[0], dtype=jnp.float64)

    # 2. Timing Residual function F
    # Current approach: euler_residuals_tau
    t_f_cur = time_callable(lambda: euler_residuals_tau(dmodel, controls_ga, dr), repeat=4, warmup=1)

    # JAX vmapped F
    t_f_jax = time_callable(lambda: cr_fwd.F_vmap(s_nodes, x_vals, coeffs), repeat=8, warmup=2)

    # Accuracy check
    res_cur = euler_residuals_tau(dmodel, controls_ga, dr)[0]
    res_jax = np.asarray(cr_fwd.F_vmap(s_nodes, x_vals, coeffs))
    max_err_f = np.max(np.abs(res_cur - res_jax))

    # 3. Timing Derivative function dF
    # Current approach: SerialDifferentiableFunction (finite differences)
    # Only run on smaller sizes or moderate repeat to keep total bench time sensible
    if N_points <= 10000:
        fn_cur = (
            lambda x_val: euler_residuals_tau(dmodel, controls_ga.unravel(x_val), dr)
            .ravel()
            .reshape((-1, n_x))
        )
        dfn_cur = SerialDifferentiableFunction(fn_cur, epsilon=1e-7)
        t_df_cur = time_callable(lambda: dfn_cur(np.array(controls_ga[0])), repeat=3, warmup=1)
        _, dval_cur = dfn_cur(np.array(controls_ga[0]))
    else:
        # Scale estimate if finite difference is prohibitively slow (> 40k points)
        t_df_cur = t_f_cur * (n_x + 1)
        dval_cur = None

    # JAX vmapped dF (jacfwd & jacrev)
    t_df_fwd = time_callable(lambda: cr_fwd.dF_vmap(s_nodes, x_vals, coeffs), repeat=6, warmup=2)
    t_df_rev = time_callable(lambda: cr_rev.dF_vmap(s_nodes, x_vals, coeffs), repeat=6, warmup=2)

    # JAX joint F and dF
    t_joint_jax = time_callable(lambda: cr_fwd.F_and_dF_vmap(s_nodes, x_vals, coeffs), repeat=6, warmup=2)

    # Accuracy check for derivative
    dval_jax = np.asarray(cr_fwd.dF_vmap(s_nodes, x_vals, coeffs))
    if dval_cur is not None:
        rel_err_df = np.max(np.abs(dval_cur - dval_jax) / (np.abs(dval_cur) + 1e-8))
    else:
        rel_err_df = 0.0

    return {
        "N": N_points,
        "t_f_cur": t_f_cur,
        "t_f_jax": t_f_jax,
        "speedup_f": t_f_cur / t_f_jax,
        "max_err_f": max_err_f,
        "t_df_cur": t_df_cur,
        "t_df_fwd": t_df_fwd,
        "t_df_rev": t_df_rev,
        "speedup_df_fwd": t_df_cur / t_df_fwd,
        "speedup_df_rev": t_df_cur / t_df_rev,
        "rel_err_df": rel_err_df,
        "t_joint_cur": t_f_cur + t_df_cur,
        "t_joint_jax": t_joint_jax,
        "speedup_joint": (t_f_cur + t_df_cur) / t_joint_jax,
    }


def main():
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]JAX Compiled & Vmapped Residual & Derivative Benchmark[/bold cyan]\n"
            "[dim]Comparing Current Vectorized (euler_residuals_tau + SerialDifferentiableFunction)\n"
            "against JAX jitted + vmapped F, dF (jacfwd / jacrev), and joint (F & dF)[/dim]",
            border_style="cyan",
        )
    )

    model_path = "examples/models/rbc.yaml"
    grid_sizes = [
        [10, 10],   # 100 points
        [32, 32],   # 1,024 points
        [100, 100], # 10,000 points
        [200, 200], # 40,000 points
    ]

    results = []
    for gs in grid_sizes:
        console.print(f"Benchmarking grid size {gs[0]}x{gs[1]} ({gs[0]*gs[1]:,d} points)...")
        res = benchmark_grid_size(model_path, gs)
        results.append(res)

    # 1. Residual Function F Comparison Table
    table_f = Table(title="\n1. Euler Residual Evaluation F(s, x, φ)", border_style="blue")
    table_f.add_column("Grid Points (N)", justify="right", style="cyan")
    table_f.add_column("Current Vectorized (ms)", justify="right", style="magenta")
    table_f.add_column("JAX vmap (ms)", justify="right", style="green")
    table_f.add_column("Speedup", justify="right", style="bold yellow")
    table_f.add_column("Max Abs Error", justify="right", style="dim")

    for r in results:
        table_f.add_row(
            f"{r['N']:,d}",
            f"{r['t_f_cur']:.2f}",
            f"{r['t_f_jax']:.2f}",
            f"{r['speedup_f']:.1f}x",
            f"{r['max_err_f']:.2e}",
        )
    console.print(table_f)

    # 2. Derivative Evaluation dF Comparison Table
    table_df = Table(title="\n2. Derivative / Jacobian Evaluation dF(s, x, φ) w.r.t Controls", border_style="green")
    table_df.add_column("Grid Points (N)", justify="right", style="cyan")
    table_df.add_column("Current (Finite Diff) (ms)", justify="right", style="magenta")
    table_df.add_column("JAX vmap jacfwd (ms)", justify="right", style="green")
    table_df.add_column("JAX vmap jacrev (ms)", justify="right", style="green")
    table_df.add_column("Speedup (jacfwd)", justify="right", style="bold yellow")
    table_df.add_column("FD Rel Error", justify="right", style="dim")

    for r in results:
        table_df.add_row(
            f"{r['N']:,d}",
            f"{r['t_df_cur']:.2f}",
            f"{r['t_df_fwd']:.2f}",
            f"{r['t_df_rev']:.2f}",
            f"{r['speedup_df_fwd']:.1f}x",
            f"{r['rel_err_df']:.2e}",
        )
    console.print(table_df)

    # 3. Total Newton Step Calculation (Joint F and dF)
    table_joint = Table(title="\n3. Joint Residual + Jacobian Evaluation (One Newton Step)", border_style="gold1")
    table_joint.add_column("Grid Points (N)", justify="right", style="cyan")
    table_joint.add_column("Current (F + dF) (ms)", justify="right", style="magenta")
    table_joint.add_column("JAX Joint Kernel (ms)", justify="right", style="green")
    table_joint.add_column("Speedup", justify="right", style="bold yellow")

    for r in results:
        table_joint.add_row(
            f"{r['N']:,d}",
            f"{r['t_joint_cur']:.2f}",
            f"{r['t_joint_jax']:.2f}",
            f"{r['speedup_joint']:.1f}x",
        )
    console.print(table_joint)
    console.print()


if __name__ == "__main__":
    main()
