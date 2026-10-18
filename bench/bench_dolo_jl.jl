using Dolo
using BenchmarkTools

root_dir = pkgdir(Dolo)
model = include("$(root_dir)/examples/ymodels/rbc_iid.jl")

grid_sizes = [
    [10, 10],
    [32, 32],
    [100, 100],
    [200, 200]
]

results = []

for gs in grid_sizes
    N = gs[1] * gs[2]
    println("Benchmarking Dolo.jl for grid $gs ($N points)...")
    dm = Dolo.discretize(model, Dict(:endo=>gs))
    wk = Dolo.time_iteration_workspace(dm)
    (;x0, φ) = wk

    # Warmup F
    Dolo.F(dm, x0, φ)
    # Benchmark F
    t_f = @belapsed Dolo.F($dm, $x0, $φ) samples=10 evals=1

    # In-place F!
    r0 = Dolo.F(dm, x0, φ)
    t_f_inplace = @belapsed Dolo.F!($r0, $dm, $x0, $φ) samples=10 evals=1

    # Warmup dF_1
    Dolo.dF_1(dm, x0, φ)
    # Benchmark dF_1
    t_df = @belapsed Dolo.dF_1($dm, $x0, $φ) samples=10 evals=1

    # In-place dF_1!
    dF0 = Dolo.dF_1(dm, x0, φ)
    t_df_inplace = @belapsed Dolo.dF_1!($dF0, $dm, $x0, $φ) samples=10 evals=1

    push!(results, Dict(
        "N" => N,
        "t_f_ms" => t_f * 1000.0,
        "t_f_inplace_ms" => t_f_inplace * 1000.0,
        "t_df_ms" => t_df * 1000.0,
        "t_df_inplace_ms" => t_df_inplace * 1000.0
    ))
end

open("bench/dolo_jl_results.json", "w") do f
    println(f, "[")
    for (i, r) in enumerate(results)
        sep = i < length(results) ? "," : ""
        n_val = r["N"]
        tf = r["t_f_ms"]
        tfi = r["t_f_inplace_ms"]
        tdf = r["t_df_ms"]
        tdfi = r["t_df_inplace_ms"]
        println(f, "  {\"N\": $n_val, \"t_f_ms\": $tf, \"t_f_inplace_ms\": $tfi, \"t_df_ms\": $tdf, \"t_df_inplace_ms\": $tdfi}$sep")
    end
    println(f, "]")
end
println("Dolo.jl benchmark completed! Results written to bench/dolo_jl_results.json")
