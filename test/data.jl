using CounterfactualExplanations
using CounterfactualExplanations.Data

@testset "Artifacts" begin
    @testset "ucr_data" begin
        @test !isnothing(CounterfactualExplanations.Data.ucr_data())
    end
    @testset "ucr_model" begin
        @test !isnothing(CounterfactualExplanations.Data.ucr_model())
    end
    @testset "cats_dogs_data" begin
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs_data())
    end
end

@testset "Toy data" begin
    @testset "toy_data_linear" begin
        @test length(toy_data_linear()) == 2
    end
    @testset "toy_data_non_linear" begin
        @test length(toy_data_non_linear()) == 2
    end
end