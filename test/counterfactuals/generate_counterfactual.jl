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
        flat_ce = CounterfactualExplanations.flatten(ce)
        @test flat_ce isa FlattenedCE
        target_encoded(flat_ce, ce.data)
        

        @testset "Unflattened" begin
            _ce = unflatten(flat_ce, ce.data, ce.M, ce.generator)
            @test _ce isa CounterfactualExplanation
            @test converged(ce) == converged(_ce)
            @test ce.x′ == _ce.x′
            @test ce.s′ == _ce.s′
            @test ce.factual == _ce.factual
            @test ce.counterfactual == _ce.counterfactual
            @test ce.target == _ce.target
            @test ce.data == _ce.data
            @test ce.M == _ce.M
            @test ce.generator == _ce.generator
            @test ce.counterfactual_state == _ce.counterfactual_state
            @test ce.target_encoded == _ce.target_encoded
            @test converged(ce) == converged(_ce)
        end
    end
end
