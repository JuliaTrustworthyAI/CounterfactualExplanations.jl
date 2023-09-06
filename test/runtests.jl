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
using PythonCall
using Printf
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.DataPreprocessing

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
    @testset "Data" begin
        include("data/data.jl")
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

    @testset "Parallelization" begin
        include("parallelization/parallelization.jl")
    end
end
