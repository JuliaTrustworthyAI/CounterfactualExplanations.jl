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
using LaplaceRedux
using LinearAlgebra
using MLDatasets
using MLJBase
using MLJDecisionTreeInterface
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
