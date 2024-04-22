using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Generators
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLUtils
using Random

# Using pretrained models is currently supported only for Julia versions >= 1.8
# This issue will be resolved as soon as possible
if VERSION >= v"1.8"
    models = _load_pretrained_models()
    @testset "Pretrained models" begin
        for (key, value) in models
            # The name of the dataset the model was trained on
            dataset_name = string(key)
            @testset "$dataset_name" begin
                for (name, M) in value[:models]
                    name = string(name)
                    @testset "$name" begin
                        data = TaijaData.data_catalogue[:vision][key]()
                        counterfactual_data = CounterfactualExplanations.DataPreprocessing.CounterfactualData(
                            data[1], data[2]
                        )
                        X = counterfactual_data.X
                        # Randomly selected factual:
                        Random.seed!(123)
                        x = select_factual(counterfactual_data, rand(1:size(X, 2)))
                        # Choose target:
                        y = Models.predict_label(M, counterfactual_data, x)
                        target = get_target(counterfactual_data, y[1])
                        # Using gravitational generator as a good representative of a generator that doesn't require latent space
                        generator = Generators.GravitationalGenerator()
                        # Single sample:
                        counterfactual = generate_counterfactual(
                            x, target, counterfactual_data, M, generator
                        )

                        @testset "Predetermined outputs" begin
                            if generator.latent_space
                                @test counterfactual.generator.latent_space
                            end
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
                            @testset "Non-trivial case, no latent space" begin
                                counterfactual_data.input_encoder = nothing
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
                                @test !Convergence.converged(
                                    counterfactual.convergence, counterfactual
                                ) || target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                @test !Convergence.converged(
                                    counterfactual.convergence, counterfactual
                                ) || length(path(counterfactual)) <= max_iter
                            end

                            @testset "Trivial case (already in target class)" begin
                                counterfactual_data.input_encoder = nothing
                                # Already in target and exceeding threshold probability:
                                y = Models.predict_label(M, counterfactual_data, x)
                                target = y[1]
                                γ = minimum([1 / length(counterfactual_data.y_levels), 0.5])
                                counterfactual = generate_counterfactual(
                                    x,
                                    target,
                                    counterfactual_data,
                                    M,
                                    generator;
                                    convergence=Convergence.DecisionThresholdConvergence(;
                                        decision_threshold=γ
                                    ),
                                    initialization=:identity,
                                )
                                x′ = CounterfactualExplanations.decode_state(counterfactual)
                                @test isapprox(counterfactual.x, x′; atol=1e-6)
                                @test Convergence.converged(
                                    counterfactual.convergence, counterfactual
                                )
                                @test CounterfactualExplanations.terminated(counterfactual)
                                @test CounterfactualExplanations.total_steps(
                                    counterfactual
                                ) == 0
                            end

                            @testset "Non-trivial case, latent space enabled" begin
                                # Using REVISE generator as a good representative of a generator that requires latent space
                                generator = Generators.REVISEGenerator()
                                for (name, vae) in value[:latent]
                                    name = string(name)
                                    @testset "$name" begin
                                        counterfactual_data.input_encoder = vae
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
                                        using CounterfactualExplanations:
                                            counterfactual_probability
                                        @test !Convergence.converged(
                                            counterfactual.convergence, counterfactual
                                        ) || target_probs(counterfactual)[1] >= γ # either not converged or threshold reached
                                        @test !Convergence.converged(
                                            counterfactual.convergence, counterfactual
                                        ) || length(path(counterfactual)) <= max_iter
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
