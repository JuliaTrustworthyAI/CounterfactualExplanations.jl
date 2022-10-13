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

# Data:
Random.seed!(1234)
xs, ys = Data.toy_data_linear()
X = hcat(xs...)
counterfactual_data = CounterfactualData(X,ys')
x = select_factual(counterfactual_data,rand(1:size(X)[2]))