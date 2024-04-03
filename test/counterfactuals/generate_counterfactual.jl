@testset "Counterfactual generation" begin
    @testset "With timer" begin
        ce = generate_counterfactual(
            x, target, counterfactual_data, M, generator; timeout=0.0001
        )
        @test typeof(ce) <: CounterfactualExplanation
    end
    @testset "Tuple input" begin
        ce = generate_counterfactual((x,), target, counterfactual_data, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end
end
