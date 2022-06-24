using CounterfactualExplanations
using Test
using Random
Random.seed!(0)

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

end
