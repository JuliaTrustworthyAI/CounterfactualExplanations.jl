using BenchmarkTools
using CounterfactualExplanations
using CounterfactualExplanations.Generators: grad_search_opt, grad_loss, grad_pen
using TaijaData: load_linearly_separable
using Random

Random.seed!(42)

# ---------------------------------------------------------------------------
# Setup: linearly-separable data, linear model, GenericGenerator
# ---------------------------------------------------------------------------
# Data
counterfactual_data = CounterfactualData(load_linearly_separable()...)

# Linear model
M_linear = fit_model(counterfactual_data, :Linear)

# MLP model
M_mlp = fit_model(counterfactual_data, :MLP; n_hidden=10, n_layers=2)

# Factual and target
target = 2
factual = 1
chosen = rand(findall(predict_label(M_linear, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)

# Generator
generator = GenericGenerator()

# Build a CounterfactualExplanation so we have a `ce` object to work with
ce_linear = generate_counterfactual(x, target, counterfactual_data, M_linear, generator)
ce_mlp    = generate_counterfactual(x, target, counterfactual_data, M_mlp, generator)

# Pre-compute an initial gradient state for apply_mutability benchmarks
_grad_state = zero(ce_linear.counterfactual_state)

# ---------------------------------------------------------------------------
# SUITE definition
# ---------------------------------------------------------------------------
SUITE = BenchmarkGroup()

# --- End-to-end benchmarks ---
SUITE["e2e"] = BenchmarkGroup(["end-to-end", "full search"])
SUITE["e2e"]["linear"] = @benchmarkable(
    generate_counterfactual($x, $target, $counterfactual_data, $M_linear, $generator),
    samples=50,
    seconds=30.0,
)
SUITE["e2e"]["mlp"] = @benchmarkable(
    generate_counterfactual($x, $target, $counterfactual_data, $M_mlp, $generator),
    samples=20,
    seconds=60.0,
)

# --- Per-iteration benchmarks ---
SUITE["per_iteration"] = BenchmarkGroup(["per-iteration", "inner-loop"])

SUITE["per_iteration"]["update!"] = @benchmarkable(
    CounterfactualExplanations.update!($ce_linear),
    samples=200,
)
SUITE["per_iteration"]["generate_perturbations"] = @benchmarkable(
    generate_perturbations($generator, $ce_linear),
    samples=200,
)
SUITE["per_iteration"]["grad_search_opt"] = @benchmarkable(
    grad_search_opt($generator, $ce_linear),
    samples=200,
)
SUITE["per_iteration"]["grad_loss"] = @benchmarkable(
    grad_loss($generator, $ce_linear),
    samples=200,
)
SUITE["per_iteration"]["grad_pen"] = @benchmarkable(
    grad_pen($generator, $ce_linear),
    samples=200,
)
SUITE["per_iteration"]["decode_state"] = @benchmarkable(
    CounterfactualExplanations.decode_state($ce_linear),
    samples=200,
)
SUITE["per_iteration"]["apply_mutability"] = @benchmarkable(
    CounterfactualExplanations.apply_mutability($ce_linear, $_grad_state),
    samples=200,
)
