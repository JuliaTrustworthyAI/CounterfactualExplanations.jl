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
init_perturbation = 2.0

models = _load_pretrained_models()

@testset "Pretrained models" begin
    generator = GravitationalGenerator()
    for (key, value) in models
        # The name of the dataset the model was trained on
        name = string(key)
        @testset "$name" begin
            for (name, M) in value[:models]
                name = string(name)
                @testset "$name" begin
                    # Randomly selected factual:
                    Random.seed!(123)
                    x = select_factual(counterfactual_data, rand(1:size(X, 2)))
                    multiple_x = select_factual(
                        counterfactual_data, rand(1:size(X, 2), 5)
                    )
                    # Choose target:
                    y = predict_label(M, counterfactual_data, x)
                    target = get_target(counterfactual_data, y[1])
                    # Single sample:
                    counterfactual = generate_counterfactual(
                        x, target, counterfactual_data, M, generator
                    )

                    @testset "Predetermined outputs" begin
                        if generator.latent_space
                            @test counterfactual.params[:latent_space]
                        end
                        @test counterfactual.target == target
                        @test counterfactual.x == x &&
                            CounterfactualExplanations.factual(counterfactual) ==
                                x
                        @test CounterfactualExplanations.factual_label(
                            counterfactual
                        ) == y
                        @test CounterfactualExplanations.factual_probability(
                            counterfactual
                        ) == probs(M, x)
                    end

                    @testset "Convergence" begin
                        @testset "Non-trivial case, no latent space" begin
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
                                max_iter=max_iter,
                                decision_threshold=γ,
                            )
                            using CounterfactualExplanations:
                                counterfactual_probability
                            @test !converged(counterfactual) ||
                                target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                            @test !converged(counterfactual) ||
                                length(path(counterfactual)) <= max_iter
                        end

                        @testset "Non-trivial case, latent space enabled" begin
                            for (name, vae) in value[:latent]
                                name = string(name)
                                @testset "$name" begin
                                    counterfactual_data.generative_model = vae
                                    # Threshold reached if converged:
                                    γ = 0.9
                                    max_iter = 1000
                                    counterfactual = generate_counterfactual(
                                        x,
                                        target,
                                        counterfactual_data,
                                        M,
                                        generator;
                                        max_iter=max_iter,
                                        decision_threshold=γ,
                                    )
                                    using CounterfactualExplanations:
                                        counterfactual_probability
                                    @test !converged(counterfactual) ||
                                        target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                    @test !converged(counterfactual) ||
                                        length(path(counterfactual)) <= max_iter
                                end
                            end
                        end

                        @testset "Trivial case (already in target class)" begin
                            counterfactual_data.generative_model = nothing
                            # Already in target and exceeding threshold probability:
                            y = predict_label(M, counterfactual_data, x)
                            target = y[1]
                            γ = minimum([
                                1 / length(counterfactual_data.y_levels), 0.5
                            ])
                            counterfactual = generate_counterfactual(
                                x,
                                target,
                                counterfactual_data,
                                M,
                                generator;
                                decision_threshold=γ,
                            )
                            @test maximum(
                                abs.(
                                    counterfactual.x .-
                                    CounterfactualExplanations.decode_state(
                                        counterfactual
                                    )
                                ),
                            ) < init_perturbation
                            @test converged(counterfactual)
                            @test CounterfactualExplanations.terminated(
                                counterfactual
                            )
                            @test CounterfactualExplanations.total_steps(
                                counterfactual
                            ) == 0
                        end
                    end
                end
            end
        end
    end
end
