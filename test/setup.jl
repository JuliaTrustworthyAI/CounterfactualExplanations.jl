using Chain: @chain
using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using Test
using DataFrames
using DecisionTree
using Flux
# using LaplaceRedux
using LinearAlgebra
using MLDatasets
using MLJBase
using Printf
using MLUtils
using Random
using TaijaData

Random.seed!(0)

using Logging
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
if is_logging(stderr)
    global_logger(NullLogger())
end

include("utils.jl")

# Load synthetic data, models, generators
synthetic = _load_synthetic()
generators = Generators.generator_catalogue

# Counteractual data and model:
counterfactual_data = CounterfactualData(load_linearly_separable()...)
M = fit_model(counterfactual_data, :Linear)
target = 2
factual = 1
chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
x = select_factual(counterfactual_data, chosen)

# Search:
generator = Generators.GenericGenerator()
ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
