using Test
using CounterfactualExplanations
using CounterfactualExplanations.Data
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Objectives

@testset "ProbeGenerator" begin
    @testset "Default arguments" begin
        generator = ProbeGenerator()
        @test typeof(generator) <: AbstractGenerator
        @test generator.λ == 0.1
        @test generator.loss == Flux.Losses.logitbinarycrossentropy
    end

    @testset "Custom arguments" begin
        generator = ProbeGenerator(; λ=0.5, loss=:mse)
        @test generator.λ == 0.5
        @test generator.loss == Flux.Losses.mse
    end
end

@testset "invalidation_rate" begin
    @testset "Invalidation rate calculation" begin
        counterfactual_data = load_linearly_separable()
        M = fit_model(counterfactual_data, :Linear)
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
        x = select_factual(counterfactual_data, chosen)
        # Search:
        generator = ProbeGenerator()
        linear_counterfactual = generate_counterfactual(
            x,
            target,
            counterfactual_data,
            M,
            generator;
            converge_when=:invalidation_rate,
            max_iter=1000,
            invalidation_rate=0.1,
            learning_rate=0.1,
        )
        rate = invalidation_rate(linear_counterfactual)
        @test rate <= 0.1
    end
end

@testset "hinge_loss" begin
    @testset "Hinge loss calculation" begin
        counterfactual_data = load_linearly_separable()
        M = fit_model(counterfactual_data, :Linear)
        target = 2
        factual = 1
        chosen = rand(findall(predict_label(M, counterfactual_data) .== factual))
        x = select_factual(counterfactual_data, chosen)
        # Search:
        generator = ProbeGenerator()
        linear_counterfactual = generate_counterfactual(
            x,
            target,
            counterfactual_data,
            M,
            generator;
            converge_when=:invalidation_rate,
            max_iter=1000,
            invalidation_rate=0.1,
            learning_rate=0.1,
        )
        loss = hinge_loss(linear_counterfactual)
        @test loss <= 0.9
    end
end
