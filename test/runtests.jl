using CounterfactualExplanations
using Test
using Random
Random.seed!(0)

using Logging
is_logging(io) = isa(io, Base.TTY) == false || (get(ENV, "CI", nothing) == "true")
if is_logging(stderr)
    global_logger(NullLogger())
end

@testset "CounterfactualExplanations.jl" begin

    @testset "Data" begin
        include("data.jl")
    end

    @testset "Data preprocessing" begin
        include("data_preprocessing.jl")
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

end
