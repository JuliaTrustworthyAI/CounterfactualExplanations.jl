using CounterfactualExplanations

@testset "Toy data" begin
    @testset "Linear" begin
        @test length(toy_data_linear()) == 2
    end
end