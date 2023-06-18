using Chain: @chain
import CompatHelperLocal as CHL
CHL.@check()
using CounterfactualExplanations
using Test
using DataFrames
using Flux
using LinearAlgebra
using MLUtils
using Random
using Plots
using LaplaceRedux
using EvoTrees
using MLJBase
using MLJDecisionTreeInterface
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.DataPreprocessing

init_perturbation = 2.0
Random.seed!(0)

using Logging
Logging.is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
if Logging.is_logging(stderr)
    Logging.global_logger(Logging.NullLogger())
end

include("utils.jl")

# Load synthetic data, models, generators
synthetic = _load_synthetic()
generators = generator_catalogue

@testset "CounterfactualExplanations.jl" begin
    @testset "Data" begin
        include("data/data.jl")
    end

    @testset "Data preprocessing" begin
        include("data_preprocessing.jl")
    end

    @testset "Generative Models" begin
        include("generative_models.jl")
    end

    @testset "Counterfactuals" begin
        include("counterfactuals.jl")
    end

    @testset "Generators" begin
        include("generators.jl")
    end

    @testset "Probe" begin
        include("probe.jl")
    end

    @testset "Model" begin
        include("models.jl")
    end

    @testset "Plotting" begin
        include("plotting.jl")
    end

    @testset "Evaluation" begin
        include("evaluation.jl")
    end
end
