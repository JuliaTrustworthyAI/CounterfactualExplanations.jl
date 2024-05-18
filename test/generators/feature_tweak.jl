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
                models = [:DecisionTreeModel, :RandomForestModel]

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
                        data.input_encoder = nothing
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
                                data.input_encoder = nothing
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
                                data.input_encoder = nothing
                                # Already in target class:
                                y = Models.predict_label(M, data, x)
                                target = y[1]
                                γ = minimum([1 / length(data.y_levels), 0.5])
                                counterfactual = CounterfactualExplanations.generate_counterfactual(
                                    x, target, data, M, generator; initialization=:identity
                                )
                                x′ = CounterfactualExplanations.decode_state(counterfactual)
                                if counterfactual.generator.latent_space == false
                                    @test isapprox(counterfactual.x, x′; atol=1e-6)
                                    @test CounterfactualExplanations.terminated(
                                        counterfactual
                                    )
                                    @test Convergence.converged(
                                        counterfactual.convergence, counterfactual
                                    )
                                end
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

    @testset "Different objectives" begin
        value = binary_synthetic[:classification_binary]
        data = value[:data]
        X = data.X

        model = :RandomForestModel
        M = Models.fit_model(data, model)
        # Randomly selected factual:
        Random.seed!(123)
        x = DataPreprocessing.select_factual(data, rand(1:size(X, 2)))
        # Choose target:
        y = Models.predict_label(M, data, x)
        target = get_target(data, y[1])
        # Single sample:
        data.input_encoder = nothing
        counterfactual = CounterfactualExplanations.generate_counterfactual(
            x, target, data, M, generator
        )

        objectives = Dict{Symbol,Any}(
            CounterfactualExplanations.Objectives.penalties_catalogue
        )
        objectives[:penalty_vector] = [
            CounterfactualExplanations.Objectives.distance_l2,
            CounterfactualExplanations.Objectives.distance_l1,
            CounterfactualExplanations.Objectives.distance_l0,
        ]

        for (name, penalty) in objectives
            @testset "$name" begin
                generator = Generators.FeatureTweakGenerator(; penalty=penalty)
                data.input_encoder = nothing
                counterfactual = CounterfactualExplanations.generate_counterfactual(
                    x, target, data, M, generator
                )
                @test Models.predict_label(
                    M, data, CounterfactualExplanations.decode_state(counterfactual)
                )[1] == target
                @test CounterfactualExplanations.terminated(counterfactual)
                @test Convergence.converged(counterfactual.convergence, counterfactual)
            end
        end
    end

    @testset "Test for incompatible model" begin
        value = binary_synthetic[:classification_binary]
        data = value[:data]
        X = data.X

        model = :MLP
        M = Models.fit_model(data, model)
        # Randomly selected factual:
        Random.seed!(123)
        x = DataPreprocessing.select_factual(data, rand(1:size(X, 2)))
        # Choose target:
        y = Models.predict_label(M, data, x)
        target = get_target(data, y[1])
        ce = CounterfactualExplanations.generate_counterfactual(
            x, target, data, M, generator
        )
        @test !converged(ce)
    end
end
