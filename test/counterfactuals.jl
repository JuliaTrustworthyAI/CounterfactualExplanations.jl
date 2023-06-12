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

    # Feature Tweak will be tested separately
    if generator_() isa HeuristicBasedGenerator
        continue
    end

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
                                    using CounterfactualExplanations:
                                        counterfactual_probability
                                    @test !converged(counterfactual) ||
                                        target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                    @test !converged(counterfactual) ||
                                        length(path(counterfactual)) <= max_iter
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
    end
end

# The feature tweak generator has to be tested separately,
# as it doesn't apply to gradient-based models and thus,
# many tests above such as testing for convergence do not apply.
@testset "Feature tweak" begin
    generator = CounterfactualExplanations.Generators.FeatureTweakGenerator()
    # Feature tweak only applies to binary classifiers
    binary_synthetic = deepcopy(synthetic)
    delete!(binary_synthetic, :classification_multi)

    @testset "Tree-based models for synthetic data" begin
        for (key, value) in binary_synthetic
            name = string(key)
            data = value[:data]
            X = data.X
            @testset "$name" begin
                # Test Feature Tweak on both a Decision Tree and a Random Forest
                models = [:DecisionTree, :RandomForest]

                for model in models
                    name = string(model)

                    @testset "$name" begin
                        M = Models.fit_model(data, model)
                        # Randomly selected factual:
                        Random.seed!(123)
                        x = select_factual(data, rand(1:size(X, 2)))
                        # Choose target:
                        y = predict_label(M, data, x)
                        target = get_target(data, y[1])
                        # Single sample:
                        counterfactual = generate_counterfactual(
                            x,
                            target,
                            data,
                            M,
                            generator
                        )

                        @testset "Predetermined outputs" begin
                            @test counterfactual.target == target
                            @test counterfactual.x == x
                            @test CounterfactualExplanations.factual(counterfactual) == x
                            @test CounterfactualExplanations.factual_label(
                                counterfactual
                            ) == y
                            @test CounterfactualExplanations.factual_probability(
                                counterfactual
                            ) == probs(M, x)
                        end

                        @testset "Counterfactual generation" begin
                            @testset "Non-trivial case" begin
                                data.generative_model = nothing
                                counterfactual = generate_counterfactual(
                                    x, target, data, M, generator
                                )
                                @test predict_label(
                                    M,
                                    data,
                                    CounterfactualExplanations.decode_state(counterfactual),
                                )[1] == target
                                @test CounterfactualExplanations.terminated(counterfactual)
                            end

                            @testset "Trivial case (already in target class)" begin
                                data.generative_model = nothing
                                # Already in target class:
                                y = predict_label(M, data, x)
                                target = y[1]
                                γ = minimum([1 / length(data.y_levels), 0.5])
                                counterfactual = generate_counterfactual(
                                    x, target, data, M, generator
                                )
                                @test maximum(
                                    abs.(
                                        counterfactual.x .-
                                        CounterfactualExplanations.decode_state(
                                            counterfactual
                                        )
                                    ),
                                ) < init_perturbation
                                @test CounterfactualExplanations.terminated(counterfactual)
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
