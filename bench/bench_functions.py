"""
Micro/macro benchmark: legacy functions (dolang) vs ModelV2 functions (JAX).

Usage:
    pixi run --environment dev python bench/bench_functions.py          # micro (~15s)
    pixi run --environment dev python bench/bench_functions.py --macro  # + time_iteration (~90s)

Notes
-----
* The V2 backend evaluates the dynspec-parsed equations through
  ``EquationsEvaluator`` with a jnp function table, wrapped in
  ``jax.jit``/``jax.vmap`` (interpreted stage of the migration).
* On a CPU-only jaxlib, ``jax.device_put`` of a small numpy array costs
  ~50-60µs, which dominates *point* calls fed with numpy arrays (as the
  algorithms do); batched calls amortize it, and passing pre-converted
  jax arrays removes it.
"""

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from dolo import yaml_import
from dolo.compiler.model_v2 import ModelV2, yaml_compose


def bench(call, repeat=3, n=100):
    best = 1e300
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(n):
            call()
        best = min(best, (time.perf_counter() - t0) / n)
    return best * 1e6  # µs


def main():
    macro = "--macro" in sys.argv
    fname = "examples/models/rbc.yaml"

    legacy = yaml_import(fname)
    v2 = ModelV2.from_yaml(yaml_compose(open(fname).read()), filename=fname)

    cl, cv = legacy.calibration, v2.calibration
    ml, sl, xl, pl = (cl["exogenous"], cl["states"], cl["controls"], cl["parameters"])
    mv, sv, xv, pv = (cv["exogenous"], cv["states"], cv["controls"], cv["parameters"])
    fl, gl = legacy.functions["arbitrage"], legacy.functions["transition"]
    fv, gv = v2.functions["arbitrage"], v2.functions["transition"]

    # warm (trace & compile)
    fv(mv, sv, xv, mv, sv, xv, pv)
    fv(mv, sv, xv, mv, sv, xv, pv, diff=True)
    gv(mv, sv, xv, mv, pv)
    mj, sj, xj, pj = (jnp.asarray(a) for a in (mv, sv, xv, pv))

    rows = [
        (
            "f point  (numpy in)",
            f"{bench(lambda: fl(ml,sl,xl,ml,sl,xl,pl), n=300):8.1f}",
            f"{bench(lambda: fv(mv,sv,xv,mv,sv,xv,pv), n=300):8.1f}",
        ),
        (
            "f point  (jax in)  ",
            "       -",
            f"{bench(lambda: fv(mj,sj,xj,mj,sj,xj,pj), n=300):8.1f}",
        ),
        (
            "g point  (numpy in)",
            f"{bench(lambda: gl(ml,sl,xl,ml,pl), n=300):8.1f}",
            f"{bench(lambda: gv(mv,sv,xv,mv,pv), n=300):8.1f}",
        ),
        (
            "f diff=True (numpy)",
            f"{bench(lambda: fl(ml,sl,xl,ml,sl,xl,pl,diff=True), n=20):8.1f}",
            f"{bench(lambda: fv(mv,sv,xv,mv,sv,xv,pv,diff=True), n=20):8.1f}",
        ),
        (
            "g diff=True (numpy)",
            f"{bench(lambda: gl(ml,sl,xl,ml,pl,diff=True), n=20):8.1f}",
            f"{bench(lambda: gv(mv,sv,xv,mv,pv,diff=True), n=20):8.1f}",
        ),
    ]

    N = 10000
    Ml, Sl, Xl = np.tile(ml, (N, 1)), np.tile(sl, (N, 1)), np.tile(xl, (N, 1))
    Mv, Sv, Xv = np.tile(mv, (N, 1)), np.tile(sv, (N, 1)), np.tile(xv, (N, 1))
    Ml, Sl, Xl = (
        np.ascontiguousarray(Ml),
        np.ascontiguousarray(Sl),
        np.ascontiguousarray(Xl),
    )
    rows += [
        (
            f"f vectorized @{N}  ",
            f"{bench(lambda: fl(Ml,Sl,Xl,Ml,Sl,Xl,pl), n=10):8.1f}",
            f"{bench(lambda: fv(Mv,Sv,Xv,Mv,Sv,Xv,pv), n=10):8.1f}",
        ),
        (
            f"g vectorized @{N}  ",
            f"{bench(lambda: gl(Ml,Sl,Xl,Ml,pl), n=10):8.1f}",
            f"{bench(lambda: gv(Mv,Sv,Xv,Mv,pv), n=10):8.1f}",
        ),
    ]

    print(f"{'operation':22s} {'legacy':>10s} {'v2 (JAX)':>10s}   (µs)")
    print("-" * 52)
    for name, l, v in rows:
        print(f"{name:22s} {l:>10s} {v:>10s}")

    if macro:
        from dolo.algos.time_iteration import time_iteration

        print("\n-- macro: time_iteration (default grid) -------------------------")
        for label, model in (("legacy", legacy), ("v2", v2)):
            t0 = time.perf_counter()
            res = time_iteration(model, verbose=False)
            dt = time.perf_counter() - t0
            print(
                f"{label:8s}: {dt:6.1f}s | iterations={res.iterations} "
                f"| converged={res.x_converged}"
            )


if __name__ == "__main__":
    main()
