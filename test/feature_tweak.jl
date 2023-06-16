# The feature tweak generator has to be tested separately from the other generators,
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

    @testset "Test for errors" begin
        # Ensure that the generator throws an error when used with a non-tree-based model
        data = synthetic[:linearly_separable][:data]
        X = data.X

        x = select_factual(data, rand(1:size(X, 2)))
        # Choose target:
        y = predict_label(M, data, x)
        target = get_target(data, y[1])
        M = Models.fit_model(data, :Linear)

        @test_throws ArgumentError generate_counterfactual(
            x, target, data, M, generator
        )

        data = synthetic[:multi_class][:data]
        X = data.X

        # Ensure that the generator throws an error when used with multi-class data
        x = select_factual(data, rand(1:size(X, 2)))
        # Choose target:
        y = predict_label(M, data, x)
        target = get_target(data, y[1])
        M = Models.fit_model(data, :DecisionTree)

        @test_throws ArgumentError generate_counterfactual(
            x, target, data, M, generator
        )
    end
end
