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

### Load synthetic data, models, generators
synthetic = CounterfactualExplanations.Data.load_synthetic()
generators = generator_catalog

@testset "CounterfactualExplanations.jl" begin

    @testset "Data" begin
        include("data.jl")
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

    @testset "Model" begin
        include("models.jl")
    end

    @testset "Plotting" begin
        include("plotting.jl")
    end

end
