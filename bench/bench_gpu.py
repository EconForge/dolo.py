"""Benchmark JAX Euler residual and Jacobian on NVIDIA GPU vs CPU.

Compares execution time of F_vmap and dF_vmap on:
- JAX on CPU
- JAX on GPU (NVIDIA GeForce RTX 2080 via CUDA 12)
Across grid sizes: N = 100, 1,024, 10,000, 40,000, 100,000, 250,000.
"""

import time
import jax
import jax.numpy as jnp
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dolo import yaml_import_v2
from dolo.numeric import DiscretizedModel
from dolo.numeric.residual_jax import compile_residual

console = Console()


def time_block(fn, reps=10, warmup=2):
    for _ in range(warmup):
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for r in res:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()

    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        res = fn()
        if hasattr(res, "block_until_ready"):
            res.block_until_ready()
        elif isinstance(res, (tuple, list)):
            for r in res:
                if hasattr(r, "block_until_ready"):
                    r.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    # Return median time in milliseconds
    times.sort()
    return times[len(times) // 2] * 1000.0


def main():
    gpu_devices = jax.devices("gpu")
    cpu_devices = jax.devices("cpu")

    if not gpu_devices:
        console.print("[bold red]No GPU device found in JAX![/bold red]")
        return

    gpu_dev = gpu_devices[0]
    cpu_dev = cpu_devices[0]

    console.print(
        Panel(
            "[bold cyan]JAX GPU vs. CPU Benchmark: Residual F and Jacobian dF_1[/bold cyan]\n"
            f"[dim]Device: {gpu_dev.device_kind} | Backend: {gpu_dev.platform.upper()} (CUDA 12)\n"
            "Model: Real Business Cycle (RBC) | Precision: Float64[/dim]",
            border_style="cyan",
        )
    )

    model_path = "examples/models/rbc.yaml"
    model = yaml_import_v2(model_path)

    # Grid configurations: (n_z, n_k)
    grid_configs = [
        (10, 10),      # N = 100
        (32, 32),      # N = 1,024
        (100, 100),    # N = 10,000
        (200, 200),    # N = 40,000
        (316, 316),    # N ~ 100,000
        (500, 500),    # N = 250,000
    ]

    results_f = []
    results_df = []

    for n_z, n_k in grid_configs:
        N = n_z * n_k
        console.print(f"Benchmarking N = {N:,d} ({n_z}x{n_k})...")

        dmodel = DiscretizedModel(model, grid_options=dict(n=[n_z, n_k]))
        controls = dmodel.initial_guess()

        cr = compile_residual(dmodel, interp_method="cubic", diff_mode="fwd")
        coeffs = cr.prefilter(controls)
        s_nodes = jnp.asarray(dmodel.endo_nodes, dtype=jnp.float64)
        x_vals = jnp.asarray(controls[0], dtype=jnp.float64)

        # Place arrays on CPU
        s_cpu = jax.device_put(s_nodes, cpu_dev)
        x_cpu = jax.device_put(x_vals, cpu_dev)
        c_cpu = jax.device_put(coeffs, cpu_dev)

        # Place arrays on GPU
        s_gpu = jax.device_put(s_nodes, gpu_dev)
        x_gpu = jax.device_put(x_vals, gpu_dev)
        c_gpu = jax.device_put(coeffs, gpu_dev)

        # Repetitions
        reps_gpu = 15 if N <= 40000 else 8
        reps_cpu = 6 if N <= 40000 else 3

        # GPU timings
        t_f_gpu = time_block(lambda: cr.F_vmap(s_gpu, x_gpu, c_gpu), reps=reps_gpu)
        t_df_gpu = time_block(lambda: cr.dF_vmap(s_gpu, x_gpu, c_gpu), reps=reps_gpu)

        # CPU timings (skip dF on CPU for very large grids to save benchmark time)
        t_f_cpu = time_block(lambda: cr.F_vmap(s_cpu, x_cpu, c_cpu), reps=reps_cpu)
        if N <= 100000:
            t_df_cpu = time_block(lambda: cr.dF_vmap(s_cpu, x_cpu, c_cpu), reps=reps_cpu)
        else:
            t_df_cpu = None

        results_f.append({
            "N": N,
            "t_cpu": t_f_cpu,
            "t_gpu": t_f_gpu,
            "speedup": t_f_cpu / t_f_gpu,
        })

        results_df.append({
            "N": N,
            "t_cpu": t_df_cpu,
            "t_gpu": t_df_gpu,
            "speedup": (t_df_cpu / t_df_gpu) if t_df_cpu else None,
        })

    # Table 1: F
    t1 = Table(title="\n1. Residual Function F Evaluation (Float64)", border_style="blue")
    t1.add_column("Grid Points (N)", justify="right", style="cyan")
    t1.add_column("JAX CPU (ms)", justify="right", style="yellow")
    t1.add_column("JAX GPU (ms)", justify="right", style="bold green")
    t1.add_column("GPU Speedup", justify="right", style="bold magenta")

    for r in results_f:
        t1.add_row(
            f"{r['N']:,d}",
            f"{r['t_cpu']:.3f}",
            f"{r['t_gpu']:.3f}",
            f"{r['speedup']:.1f}x",
        )
    console.print(t1)

    # Table 2: dF
    t2 = Table(title="\n2. Derivative / Jacobian dF_1 Evaluation w.r.t Controls (Float64)", border_style="green")
    t2.add_column("Grid Points (N)", justify="right", style="cyan")
    t2.add_column("JAX CPU (ms)", justify="right", style="yellow")
    t2.add_column("JAX GPU (ms)", justify="right", style="bold green")
    t2.add_column("GPU Speedup", justify="right", style="bold magenta")

    for r in results_df:
        cpu_str = f"{r['t_cpu']:.3f}" if r['t_cpu'] is not None else "N/A (>500ms)"
        sp_str = f"{r['speedup']:.1f}x" if r['speedup'] is not None else "-"
        t2.add_row(
            f"{r['N']:,d}",
            cpu_str,
            f"{r['t_gpu']:.3f}",
            sp_str,
        )
    console.print(t2)
    console.print()


if __name__ == "__main__":
    main()
