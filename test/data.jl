@testset "Artifacts" begin

    @testset "cats_dogs" begin
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs_data())
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs_model())
        # @test !isnothing(CounterfactualExplanations.Data.cats_dogs_laplace())
    end

    @testset "Synthetic Data" begin
        @test !isnothing(CounterfactualExplanations.Data.load_synthetic([:flux]))
    end
end

@testset "Toy data" begin
    @test length(toy_data_linear()) == 2
    @test length(toy_data_non_linear()) == 2
    @test length(toy_data_multi()) == 2
end
