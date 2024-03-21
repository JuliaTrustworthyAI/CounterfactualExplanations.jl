using Chain: @chain
using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using Test
using DataFrames
using EvoTrees
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

@testset "CounterfactualExplanations.jl" begin
    include("aqua.jl")

    @testset "Data" begin
        include("data/data_preprocessing.jl")
    end

    @testset "Generators" begin
        include("generators/generators.jl")
    end

    @testset "Models" begin
        include("models/models.jl")
    end

    @testset "Evaluation" begin
        include("other/evaluation.jl")
    end

    @testset "Objectives" begin
        include("other/objectives.jl")
    end

    @testset "Parallelization" begin
        include("parallelization/parallelization.jl")
    end

    @testset "Other" begin
        include("other/other.jl")
    end
end
