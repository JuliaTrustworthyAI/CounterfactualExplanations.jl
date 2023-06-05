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

# NOTE:
# This is probably the most important/useful test script, because it runs through the whole process of: 
# - loading artifacts
# - setting up counterfactual search for various models and generators
# - running counterfactual search

# LOOP:
for (key, generator_) in generators
    name = uppercasefirst(string(key))
    @testset "$name" begin

        # Generator:
        generator = deepcopy(generator_())

        @testset "Models for synthetic data" begin
            for (key, value) in synthetic
                name = string(key)
                @testset "$name" begin
                    counterfactual_data = value[:data]
                    X = counterfactual_data.X
                    ys_cold = vec(counterfactual_data.y)

                    for (likelihood, model) in value[:models]
                        if generator isa HeuristicBasedGenerator && !(model[:model] isa TreeModel)
                            continue
                        end
                        name = string(likelihood)
                        @testset "$name" begin
                            M = model[:model]
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
                                        max_iter=max_iter,
                                        decision_threshold=γ,
                                    )
                                    # heuristic based generators don't use gradients and therefore dont check for convergence
                                    if !(generator isa HeuristicBasedGenerator)
                                        using CounterfactualExplanations:
                                            counterfactual_probability
                                        @test !converged(counterfactual) ||
                                            target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                        @test !converged(counterfactual) ||
                                            length(path(counterfactual)) <= max_iter
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
                                    if !(M isa TreeModel)
                                        @test length(path(counterfactual)) == 1
                                    end
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
    end
end
