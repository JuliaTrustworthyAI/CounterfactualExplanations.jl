using CounterfactualExplanations
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLJBase
using MLJDecisionTreeInterface
using MLUtils
using Random
using LaplaceRedux

@testset "Standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X
            for (likelihood, model) in value[:models]
                name = string(likelihood)
                @testset "$name" begin
                    @testset "Matrix of inputs" begin
                        @test size(logits(model[:model], X))[2] == size(X, 2)
                        @test size(probs(model[:model], X))[2] == size(X, 2)
                        @test model[:model].likelihood == value[:data].likelihood
                    end
                    @testset "Vector of inputs" begin
                        @test size(logits(model[:model], X[:, 1]), 2) == 1
                        @test size(probs(model[:model], X[:, 1]), 2) == 1
                        @test model[:model].likelihood == value[:data].likelihood
                    end
                end
            end
        end
    end
end

@testset "Non-standard models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            # Test the EvoTree model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :EvoTree)
            X = value[:data].X
            name = "EvoTree"

            @testset "$name" begin
                @test model.likelihood == value[:data].likelihood
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the DecisionTree model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :DecisionTree)
            name = "DecisionTree"
            @testset "$name" begin
                @test model.likelihood == value[:data].likelihood
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the RandomForest model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :RandomForest)
            name = "RandomForest"
            @testset "$name" begin
                @test model.likelihood == value[:data].likelihood
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                end
            end

            # Test the LaplaceReduxModel
            flux_model = CounterfactualExplanations.Models.fit_model(value[:data], :Linear).model
            laplace_model = LaplaceRedux.Laplace(flux_model; likelihood=:classification)
            model = Models.LaplaceReduxModel(laplace_model; likelihood=:classification_binary)

            name = "LaplaceRedux"
            @testset "$name" begin
                @test model.likelihood == :classification_binary
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                end
            end
        end
    end
end

@testset "Test for errors" begin
    @test_throws ArgumentError Models.FluxModel("dummy"; likelihood=:regression)
    @test_throws ArgumentError Models.FluxEnsemble("dummy"; likelihood=:regression)
    @test_throws ArgumentError Models.LaplaceReduxModel("dummy"; likelihood=:regression)
    @test_throws ArgumentError Models.EvoTreeModel("dummy"; likelihood=:regression)

    data = CounterfactualExplanations.Data.load_linearly_separable()
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    M = MLJDecisionTreeInterface.DecisionTreeClassifier()
    tree_model = MLJBase.machine(M, X, y)

    @test_throws ArgumentError Models.TreeModel(tree_model; likelihood=:regression)

    M = MLJDecisionTreeInterface.RandomForestClassifier()
    forest_model = MLJBase.machine(M, X, y)

    @test_throws ArgumentError Models.TreeModel(forest_model; likelihood=:regression)

    M = MLJDecisionTreeInterface.DecisionTreeRegressor()
    regression_model = MLJBase.machine(M, X, y)

    @test_throws ArgumentError Models.TreeModel(
        regression_model; likelihood=:classification_binary
    )
    @test_throws ArgumentError Models.TreeModel(
        regression_model; likelihood=:classification_multi
    )

    flux_model = CounterfactualExplanations.Models.fit_model(value[:data], :Linear).model
    laplace_model = LaplaceRedux.Laplace(flux_model; likelihood=:classification)

    @test_throws ArgumentError Models.LaplaceReduxModel(
        laplace_model; likelihood=:classification_multi
    )
    @test_throws ArgumentError Models.LaplaceReduxModel(
        laplace_model; likelihood=:regression
    )
end
