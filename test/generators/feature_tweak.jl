# The feature tweak generator has to be tested separately,
# as it doesn't apply to gradient-based models and thus,
# many tests above such as testing for convergence do not apply.
@testset "Feature tweak" begin
    generator = Generators.FeatureTweakGenerator()
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
                        x = DataPreprocessing.select_factual(data, rand(1:size(X, 2)))
                        # Choose target:
                        y = Models.predict_label(M, data, x)
                        target = get_target(data, y[1])
                        # Single sample:
                        counterfactual = CounterfactualExplanations.generate_counterfactual(
                            x, target, data, M, generator
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
                            ) == Models.probs(M, x)
                        end

                        @testset "Counterfactual generation" begin
                            @testset "Non-trivial case" begin
                                data.generative_model = nothing
                                counterfactual = CounterfactualExplanations.generate_counterfactual(
                                    x, target, data, M, generator
                                )
                                @test Models.predict_label(
                                    M,
                                    data,
                                    CounterfactualExplanations.decode_state(counterfactual),
                                )[1] == target
                                @test CounterfactualExplanations.terminated(counterfactual)
                            end

                            @testset "Trivial case (already in target class)" begin
                                data.generative_model = nothing
                                # Already in target class:
                                y = Models.predict_label(M, data, x)
                                target = y[1]
                                γ = minimum([1 / length(data.y_levels), 0.5])
                                counterfactual = CounterfactualExplanations.generate_counterfactual(
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
                                ) == 1
                            end
                        end
                    end
                end
            end
        end
    end
end
