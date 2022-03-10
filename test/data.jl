using CounterfactualExplanations

@testset "Data" begin
    @testset "ucr_data" begin
        @test !isnothing(CounterfactualExplanations.Data.ucr_data())
    end
    @testset "ucr_model" begin
        @test !isnothing(CounterfactualExplanations.Data.ucr_model())
    end
    @testset "cats_dogs_data" begin
        @test !isnothing(CounterfactualExplanations.Data.cats_dogs())
    end
end