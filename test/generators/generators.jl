include("clue.jl")
include("feature_tweak.jl")
include("full_runs.jl")
include("probe.jl")
include("growing_spheres.jl")
include("categorical_data.jl")

# generic tests
@testset "Construction" begin
    @testset "Generic" begin
        generator = Generators.GenericGenerator()
        @test typeof(generator) <: Generators.AbstractGradientBasedGenerator
    end

    @testset "Macros" begin
        generator = Generators.GenericGenerator()
        @chain generator begin
            @objective logitcrossentropy + 5.0ddp_diversity
            @with_optimiser JSMADescent(η=0.5)
            @search_latent_space
        end
        @test typeof(generator.loss) <: Function
        @test typeof(generator.opt) == Generators.JSMADescent
        @test generator.latent_space
    end
end
