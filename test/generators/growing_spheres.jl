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

@testset "Growing Spheres" begin
    generator = CounterfactualExplanations.Generators.GrowingSpheresGenerator()
    models = CounterfactualExplanations.Models.standard_models_catalogue
    @testset "Models for synthetic data" begin
        for (key, value) in synthetic
            name = string(key)
            @testset "$name" begin
                counterfactual_data = value[:data]
                X = counterfactual_data.X
                # Loop over values of the dict

                for (model_name, model) in models
                    name = string(model_name)
                    @testset "$name" begin
                        M = CounterfactualExplanations.Models.fit_model(
                            counterfactual_data, model_name
                        )
                        # Randomly selected factual:
                        Random.seed!(123)
                        x = select_factual(counterfactual_data, rand(1:size(X, 2)))
                        # Choose target:
                        y = predict_label(M, counterfactual_data, x)
                        target = get_target(counterfactual_data, y[1])
                        # Single sample:
                        counterfactual = generate_counterfactual(
                            x, target, counterfactual_data, M, generator
                        )

                        @testset "Predetermined outputs" begin
                            @test counterfactual.target == target
                            @test counterfactual.x == x &&
                                CounterfactualExplanations.factual(counterfactual) == x
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
                                counterfactual = generate_counterfactual(
                                    x, target, counterfactual_data, M, generator;
                                )
                                @test Models.predict_label(
                                    M,
                                    counterfactual_data,
                                    CounterfactualExplanations.decode_state(counterfactual),
                                )[1] == target
                                @test CounterfactualExplanations.terminated(counterfactual)
                            end

                            @testset "Trivial case (already in target class)" begin
                                counterfactual_data.generative_model = nothing
                                # Already in target and exceeding threshold probability:
                                counterfactual = generate_counterfactual(
                                    x, target, counterfactual_data, M, generator
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