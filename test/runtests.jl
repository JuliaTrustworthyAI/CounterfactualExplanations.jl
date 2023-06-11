using Chain: @chain
import CompatHelperLocal as CHL
CHL.@check()
using CounterfactualExplanations
using Random
using Test

Random.seed!(0)

using Logging
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
if is_logging(stderr)
    global_logger(NullLogger())
end

include("utils.jl")

# Load synthetic data, models, generators
synthetic = _load_synthetic()
generators = generator_catalogue

@testset "CounterfactualExplanations.jl" begin
    # @testset "Data" begin
    #     include("data.jl")
    # end

    # @testset "Data preprocessing" begin
    #     include("data_preprocessing.jl")
    # end

    # @testset "Generative Models" begin
    #     include("generative_models.jl")
    # end

    @testset "Counterfactuals" begin
        include("counterfactuals.jl")
    end

    # @testset "Generators" begin
    #     include("generators.jl")
    # end

    # @testset "Probe" begin
    #     include("probe.jl")
    # end

    # @testset "Model" begin
    #     include("models.jl")
    # end

    # @testset "Plotting" begin
    #     include("plotting.jl")
    # end

    # @testset "Evaluation" begin
    #     include("evaluation.jl")
    # end
end
