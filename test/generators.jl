using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using Random

@testset "Construction" begin
    @testset "Generic" begin
        generator = GenericGenerator()
        @test typeof(generator) <: AbstractGradientBasedGenerator
    end

    @testset "Macros" begin
        generator = GenericGenerator()
        @chain generator begin
            @objective logitcrossentropy + 5.0ddp_diversity
            @with_optimiser JSMADescent(η=0.5)
            @search_latent_space
        end
        @test typeof(generator.loss) <: Function
        @test typeof(generator.opt) == JSMADescent
        @test generator.latent_space
    end
end
