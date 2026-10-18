"""
Side-by-side Benchmark: Dolo.jl vs. dolo.py (JAX vmap) vs. dolo.py (Current Vectorized).

Same RBC model, identical grid points and parameters:
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

from dolo import yaml_import_v2
from dolo.numeric import DiscretizedModel
from dolo.algos.time_iteration import euler_residuals_tau
from dolo.numeric.optimize.newton import SerialDifferentiableFunction
from dolo.numeric.residual_jax import compile_residual

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
    grid_sizes = [
        [10, 10],    # 100
        [32, 32],    # 1,024
        [100, 100],  # 10,000
        [200, 200],  # 40,000
    ]

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]Cross-Platform Benchmark: dolo.py (JAX) vs. Dolo.jl vs. dolo.py (Current)[/bold cyan]\n"
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
        model = yaml_import_v2(model_path)
        dmodel = DiscretizedModel(model, grid_options=dict(n=gs))
        controls_ga = dmodel.initial_guess()
        dr = dmodel.decision_rule(controls_ga)
        n_x = len(dmodel.model.calibration["controls"])

        cr = compile_residual(dmodel, interp_method="cubic", diff_mode="fwd")
        coeffs = cr.prefilter(controls_ga)
        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        x_vals = jnp.asarray(controls_ga[0], dtype=jnp.float64)

        # F timings
        t_f_cur = time_callable(lambda: euler_residuals_tau(dmodel, controls_ga, dr), repeat=4, warmup=1)
        t_f_jax = time_callable(lambda: cr.F_vmap(s_nodes, x_vals, coeffs), repeat=8, warmup=2)
        t_f_jl = jl_results[N]["t_f_inplace_ms"]

        # dF timings
        if N <= 10000:
            fn_cur = (
                lambda x_val: euler_residuals_tau(dmodel, controls_ga.unravel(x_val), dr)
                .ravel()
                .reshape((-1, n_x))
            )
            dfn_cur = SerialDifferentiableFunction(fn_cur, epsilon=1e-7)
            t_df_cur = time_callable(lambda: dfn_cur(np.array(controls_ga[0])), repeat=3, warmup=1)
        else:
            t_df_cur = t_f_cur * (n_x + 1)

        t_df_jax = time_callable(lambda: cr.dF_vmap(s_nodes, x_vals, coeffs), repeat=6, warmup=2)
        t_df_jl = jl_results[N]["t_df_inplace_ms"]

        rows_f.append({
            "N": N,
            "t_cur": t_f_cur,
            "t_jl": t_f_jl,
            "t_jax": t_f_jax,
            "speedup_vs_cur": t_f_cur / t_f_jax,
            "ratio_vs_jl": t_f_jl / t_f_jax,
        })

        rows_df.append({
            "N": N,
            "t_cur": t_df_cur,
            "t_jl": t_df_jl,
            "t_jax": t_df_jax,
            "speedup_vs_cur": t_df_cur / t_df_jax,
            "ratio_vs_jl": t_df_jl / t_df_jax,
        })

    # Table 1: Residual F
    t1 = Table(title="\n1. Residual Function F Evaluation", border_style="blue")
    t1.add_column("Grid Points (N)", justify="right", style="cyan")
    t1.add_column("dolo.py Current (ms)", justify="right", style="magenta")
    t1.add_column("Dolo.jl (ms)", justify="right", style="bright_yellow")
    t1.add_column("dolo.py JAX vmap (ms)", justify="right", style="bold green")
    t1.add_column("Speedup vs Current", justify="right", style="green")
    t1.add_column("JAX vs Dolo.jl", justify="right", style="bold cyan")

    for r in rows_f:
        cmp_str = f"{r['ratio_vs_jl']:.2f}x faster" if r['ratio_vs_jl'] >= 1.0 else f"{1.0/r['ratio_vs_jl']:.2f}x slower"
        t1.add_row(
            f"{r['N']:,d}",
            f"{r['t_cur']:.2f}",
            f"{r['t_jl']:.2f}",
            f"{r['t_jax']:.2f}",
            f"{r['speedup_vs_cur']:.1f}x",
            cmp_str,
        )
    console.print(t1)

    # Table 2: Derivative dF
    t2 = Table(title="\n2. Derivative / Jacobian dF_1 Evaluation w.r.t Controls", border_style="green")
    t2.add_column("Grid Points (N)", justify="right", style="cyan")
    t2.add_column("dolo.py Current FD (ms)", justify="right", style="magenta")
    t2.add_column("Dolo.jl ForwardDiff (ms)", justify="right", style="bright_yellow")
    t2.add_column("dolo.py JAX jacfwd (ms)", justify="right", style="bold green")
    t2.add_column("Speedup vs Current", justify="right", style="green")
    t2.add_column("JAX vs Dolo.jl", justify="right", style="bold cyan")

    for r in rows_df:
        cmp_str = f"{r['ratio_vs_jl']:.2f}x faster" if r['ratio_vs_jl'] >= 1.0 else f"{1.0/r['ratio_vs_jl']:.2f}x slower"
        t2.add_row(
            f"{r['N']:,d}",
            f"{r['t_cur']:.2f}",
            f"{r['t_jl']:.2f}",
            f"{r['t_jax']:.2f}",
            f"{r['speedup_vs_cur']:.1f}x",
            cmp_str,
        )
    console.print(t2)
    console.print()


if __name__ == "__main__":
    main()
