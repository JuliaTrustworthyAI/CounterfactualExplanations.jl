using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Data
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLUtils
using Random

@testset "CLUE" begin
    generator = CLUEGenerator()

    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            counterfactual_data = value[:data]
            X = counterfactual_data.X
            ys_cold = vec(counterfactual_data.y)

            for (likelihood, model) in value[:models]
                name = string(likelihood)
                @testset "$name" begin
                    M = model[:model]
                    # Randomly selected factual:
                    Random.seed!(123)
                    x = select_factual(counterfactual_data, rand(1:size(X, 2)))
                    multiple_x = select_factual(counterfactual_data, rand(1:size(X, 2), 5))
                    # Choose target:
                    y = predict_label(M, counterfactual_data, x)
                    target = get_target(counterfactual_data, y[1])
                    # Single sample:
                    counterfactual = generate_counterfactual(
                        x, target, counterfactual_data, M, generator
                    )
                    # Multiple samples:
                    counterfactuals = generate_counterfactual(
                        multiple_x, target, counterfactual_data, M, generator
                    )

                    @testset "Predetermined outputs" begin
                        if generator.latent_space
                            @test counterfactual.params[:latent_space]
                        end
                        @test counterfactual.target == target
                        @test counterfactual.x == x &&
                            CounterfactualExplanations.factual(counterfactual) == x
                        @test CounterfactualExplanations.factual_label(counterfactual) == y
                        @test CounterfactualExplanations.factual_probability(
                            counterfactual
                        ) == probs(M, x)
                    end

                    @testset "Convergence" begin
                        @testset "Non-trivial case" begin
                            counterfactual_data.generative_model = nothing
                            # Threshold reached if converged:
                            γ = 0.9
                            max_iter = 1000
                            counterfactual = generate_counterfactual(
                                x,
                                target,
                                counterfactual_data,
                                M,
                                generator;
                                convergence=Convergence.DecisionThresholdConvergence(;
                                    max_iter=max_iter, decision_threshold=γ
                                ),
                            )
                            using CounterfactualExplanations: counterfactual_probability
                            @test !Convergence.converged(counterfactual.convergence, counterfactual) ||
                                target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                            @test !Convergence.converged(counterfactual.convergence, counterfactual) ||
                                length(path(counterfactual)) <= max_iter
                        end
                    end
                end
            end
        end
    end
end
