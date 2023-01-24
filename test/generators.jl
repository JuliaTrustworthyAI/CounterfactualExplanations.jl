using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using Random

@testset "Construction" begin

    @testset "Generic" begin
        generator = GenericGenerator()
        @test hasfield(GenericGenerator, :loss)
        @test typeof(generator) <: AbstractGradientBasedGenerator
    end

    @testset "Greedy" begin
        generator = GreedyGenerator()
        @test hasfield(GreedyGenerator, :loss)
        @test typeof(generator) <: AbstractGradientBasedGenerator
    end

end
