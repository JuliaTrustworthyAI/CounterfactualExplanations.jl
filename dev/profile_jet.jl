using CounterfactualExplanations
using CounterfactualExplanations.Generators: grad_search_opt, grad_loss, grad_pen
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using TaijaData: load_linearly_separable
using Random
using JET
using InteractiveUtils
using Printf

Random.seed!(42)

const OUT_DIR = joinpath(@__DIR__, "artifacts", "profile")
mkpath(OUT_DIR)

# ---------------------------------------------------------------------------
# Setup: same as benchmark/benchmarks.jl
# ---------------------------------------------------------------------------
counterfactual_data = CounterfactualData(load_linearly_separable()...)
M_linear = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M_linear, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)
generator = GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M_linear, generator)
_grad_state = zero(ce.counterfactual_state)

println("Setup complete.")
println("  ce typeof: ", typeof(ce))
println("  counterfactual_state typeof: ", typeof(ce.counterfactual_state))
println("  search typeof: ", typeof(ce.search))
println()

# ---------------------------------------------------------------------------
# Helper: capture @code_warntype output to string
# ---------------------------------------------------------------------------
function warntype_to_string(f, args...)
    # Use sprint with code_warntype (function form accepts IO)
    types = Tuple{map(typeof, args)...}
    try
        # code_warntype(io, f, types) writes to io
        return sprint(InteractiveUtils.code_warntype, f, types)
    catch e
        # Fallback: try the macro form with redirect
        try
            io = IOBuffer()
            redirect_stdout(io) do
                InteractiveUtils.@code_warntype f(args...)
            end
            return String(take!(io))
        catch e2
            return "ERROR during @code_warntype: $e2"
        end
    end
end

# ---------------------------------------------------------------------------
# Helper: capture JET.report_opt to string
# ---------------------------------------------------------------------------
function jet_opt_to_string(f, argtypes...)
    # JET.report_opt expects a single Tuple type, not vararg types
    tuple_type = Tuple{argtypes...}
    io = IOBuffer()
    try
        r = JET.report_opt(f, tuple_type)
        show(io, MIME"text/plain"(), r)
    catch e
        println(io, "ERROR during JET.report_opt: ", e)
        println(io, "Backtrace:")
        println(io, sprint(showerror, e, catch_backtrace()))
    end
    return String(take!(io))
end

# ---------------------------------------------------------------------------
# Functions to profile: (name, function, call_args, arg_types_for_jet)
# ---------------------------------------------------------------------------
profiles = [
    ("update",                CounterfactualExplanations.update!,       (ce,), (typeof(ce),)),
    ("generate_perturbations", Generators.generate_perturbations,       (generator, ce), (typeof(generator), typeof(ce))),
    ("propose_state",          Generators.propose_state,                (generator, ce), (typeof(generator), typeof(ce))),
    ("grad_search_opt",        grad_search_opt,                         (generator, ce), (typeof(generator), typeof(ce))),
    ("grad_loss",              grad_loss,                               (generator, ce), (typeof(generator), typeof(ce))),
    ("grad_pen",               grad_pen,                                (generator, ce), (typeof(generator), typeof(ce))),
    ("decode_state",           CounterfactualExplanations.decode_state, (ce,), (typeof(ce),)),
    ("apply_mutability",       CounterfactualExplanations.apply_mutability, (ce, _grad_state), (typeof(ce), typeof(_grad_state))),
]

# ---------------------------------------------------------------------------
# Warmup: make sure all functions have been called once
# ---------------------------------------------------------------------------
println("Warming up functions...")
try
    CounterfactualExplanations.update!(ce)
    Generators.generate_perturbations(generator, ce)
    Generators.propose_state(generator, ce)
    grad_search_opt(generator, ce)
    grad_loss(generator, ce)
    grad_pen(generator, ce)
    CounterfactualExplanations.decode_state(ce)
    CounterfactualExplanations.apply_mutability(ce, _grad_state)
catch e
    println("WARNING: warmup error: ", e)
end
println("Warmup done.\n")

# ---------------------------------------------------------------------------
# Run profiling
# ---------------------------------------------------------------------------
jet_results = Dict{String,String}()
warntype_results = Dict{String,String}()

for (name, f, call_args, argtypes) in profiles
    println("=" ^ 60)
    println("Profiling: ", name)
    println("=" ^ 60)

    # @code_warntype
    println("  Running @code_warntype...")
    wt_str = warntype_to_string(f, call_args...)
    warntype_results[name] = wt_str
    wt_path = joinpath(OUT_DIR, "codewarntype_$(name).txt")
    write(wt_path, wt_str)
    println("  Saved: ", wt_path)

    # JET.report_opt
    println("  Running JET.report_opt...")
    jet_str = jet_opt_to_string(f, argtypes...)
    jet_results[name] = jet_str
    jet_path = joinpath(OUT_DIR, "jet_$(name).txt")
    write(jet_path, jet_str)
    println("  Saved: ", jet_path)

    println()
end

# ---------------------------------------------------------------------------
# Print quick summary to stdout
# ---------------------------------------------------------------------------
println("\n", "=" ^ 60)
println("QUICK SUMMARY")
println("=" ^ 60)
for name in sort(collect(keys(jet_results)))
    jet_str = jet_results[name]
    has_no_issues = occursin("No issues", jet_str) || occursin("No errors", jet_str) || isempty(jet_str)
    error_count = count("┌", jet_str)
    println(@sprintf("  %-25s JET issues: %s", name, has_no_issues ? "none" : "$error_count blocks"))
end
println("\nDone. Artefacts saved to: ", OUT_DIR)
