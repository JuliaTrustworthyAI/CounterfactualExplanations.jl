# Here we test edge cases for the function 'generate_counterfactual'.
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

    @testset "Field aliases" begin
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
        @test ce.factual == ce.x
        @test ce.counterfactual == ce.x′
        @test ce.counterfactual_state == ce.s′
    end

    @testset "Flattening" begin
        ce = generate_counterfactual(x, target, counterfactual_data, M, generator)
        flat_ce = flatten(ce)
        @test flat_ce isa FlattenedCE
        target_encoded(flat_ce)
        _ce = unflatten(
            flat_ce,
            ce.data,
            ce.M,
            ce.generator
        )
        @test _ce isa CounterfactualExplanation
    end
end
