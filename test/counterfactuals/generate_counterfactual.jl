using CounterfactualExplanations.Objectives: distance_mad

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

    @testset "No penalty" begin
        generator.penalty = nothing
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end

    @testset "Keyword penalty" begin
        pen = [(distance_mad, (agg=sum,))]
        generator.penalty = pen
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
        @test typeof(ce) <: CounterfactualExplanation
    end
end
