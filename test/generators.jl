using CounterfactualExplanations
using CounterfactualExplanations.Generators

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

# Data:
using Random
using CounterfactualExplanations.Data
Random.seed!(1234)
xs, ys = Data.toy_data_linear()
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')
x = select_factual(counterfactual_data,rand(1:size(X)[2]))