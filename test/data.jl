using CounterfactualExplanations
using CounterfactualExplanations.Data
import CounterfactualExplanations.Data: toy_data_linear, toy_data_multi, toy_data_non_linear

@testset "Artifacts" begin
    
    @testset "cats_dogs" begin
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs_data())
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs_model())
        # @test !isnothing(CounterfactualExplanations.Data.cats_dogs_laplace())
    end

    @testset "MNIST" begin
        @test !isnothing(CounterfactualExplanations.Data.mnist_data())
        @test !isnothing(CounterfactualExplanations.Data.mnist_model())
        @test !isnothing(CounterfactualExplanations.Data.mnist_ensemble())
    end
end

@testset "Toy data" begin
    @test length(toy_data_linear()) == 2
    @test length(toy_data_non_linear()) == 2
    @test length(toy_data_multi()) == 2
end