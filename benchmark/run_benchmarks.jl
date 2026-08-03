using BenchmarkTools

# Load the benchmark suite definition
include("benchmarks.jl")

# Run the full suite
results = run(SUITE; verbose=true)

# Save results as baseline
baseline_path = joinpath(@__DIR__, "baseline.json")
BenchmarkTools.save(baseline_path, results)

println("Benchmark results saved to: ", baseline_path)
