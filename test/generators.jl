using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using Random

@testset "Construction" begin

    @testset "Generic" begin
        generator = GenericGenerator()
        @test typeof(generator) <: AbstractGradientBasedGenerator
    end

    @testset "Greedy" begin
        generator = GreedyGenerator()
        @test typeof(generator) <: AbstractGradientBasedGenerator
    end

end
