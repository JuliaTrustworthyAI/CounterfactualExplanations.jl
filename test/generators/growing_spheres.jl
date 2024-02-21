using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLUtils
using Random

@testset "Growing Spheres" begin
    convergence = CounterfactualExplanations.Convergence.GeneratorConditionsConvergence(max_iter=1000)
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

                        @testset "Convergence" begin
                            @testset "Non-trivial case" begin
                                generator = CounterfactualExplanations.Generators.GrowingSpheresGenerator()
                                counterfactual_data.generative_model = nothing
                                # Threshold reached if converged:
                                counterfactual = generate_counterfactual(
                                    x, target, counterfactual_data, M, generator; convergence = convergence
                                )
                                @test CounterfactualExplanations.Models.predict_label(
                                    M, counterfactual_data, counterfactual.x′
                                )[1] == target

                                @test CounterfactualExplanations.terminated(counterfactual)
                            end

                            @testset "Trivial case (already in target class)" begin
                                counterfactual_data.generative_model = nothing
                                # Already in target class:
                                y = CounterfactualExplanations.Models.predict_label(
                                    M, counterfactual_data, x
                                )
                                target = y[1]
                                γ = minimum([1 / length(counterfactual_data.y_levels), 0.5])
                                generator = CounterfactualExplanations.Generators.GrowingSpheresGenerator()
                                counterfactual = CounterfactualExplanations.generate_counterfactual(
                                    x, target, counterfactual_data, M, generator; initialization=:identity, convergence = convergence
                                )
                                x′ = counterfactual.x′
                                if counterfactual.generator.latent_space == false
                                    @test isapprox(counterfactual.x, x′; atol=1e-6)
                                end
                                @test CounterfactualExplanations.terminated(counterfactual)
                            end
                        end
                    end
                end
            end
        end
    end
end
