using CounterfactualExplanations
using Test
using Random
Random.seed!(0)

@testset "CounterfactualExplanations.jl" begin

    @testset "Generators" begin
        include("generators.jl")
    end

    @testset "Counterfactuals" begin
        include("counterfactuals.jl")
    end

    @testset "Losses" begin
        include("losses.jl")
    end

    @testset "Model" begin
        include("models.jl")
    end

    @testset "Data" begin
        include("data.jl")
    end

end
