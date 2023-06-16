using CounterfactualExplanations
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLJBase
using MLJDecisionTreeInterface
using MLUtils
using Random

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

@testset "Tree-based models for synthetic data" begin
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            # Test the EvoTree model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :EvoTree)
            X = value[:data].X
            name = "EvoTree"

            @testset "$name" begin
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                    @test model.likelihood == value[:data].likelihood
                end
            end

            # Test the DecisionTree model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :DecisionTree)
            name = "DecisionTree"
            @testset "$name" begin
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                    @test model.likelihood == value[:data].likelihood
                end
            end

            # Test the RandomForest model
            model = CounterfactualExplanations.Models.fit_model(value[:data], :RandomForest)
            name = "RandomForest"
            @testset "$name" begin
                @testset "Matrix of inputs" begin
                    @test size(logits(model, X))[2] == size(X, 2)
                    @test size(probs(model, X))[2] == size(X, 2)
                    @test model.likelihood == value[:data].likelihood
                end
                @testset "Vector of inputs" begin
                    @test size(logits(model, X[:, 1]), 2) == 1
                    @test size(probs(model, X[:, 1]), 2) == 1
                    @test model.likelihood == value[:data].likelihood
                end
            end
        end
    end
end

mutable struct Dummy end

@testset "Test for errors" begin
    dummy = Dummy()

    @test_throws ArgumentError FluxModel(dummy; likelihood=:regression)
    @test_throws ArgumentError FluxEnsemble(dummy; likelihood=:regression)
    @test_throws ArgumentError Linear(dummy; likelihood=:regression)
    @test_throws ArgumentError LaplaceReduxModel(dummy; likelihood=:regression)
    @test_throws ArgumentError EvoTreeModel(dummy; likelihood=:regression)

    data = CounterfactualExplanations.Data.load_linearly_separable()
    X, y = CounterfactualExplanations.DataPreprocessing.preprocess_data_for_mlj(data)
    M = MLJDecisionTreeInterface.DecisionTreeClassifier()
    tree_model = MLJBase.machine(M, X, y)

    @test_throws ArgumentError DecisionTreeModel(tree_model; likelihood=:regression)
    @test_throws ArgumentError DecisionTreeModel(dummy; likelihood=:classification_binary)
    @test_throws ArgumentError DecisionTreeModel(dummy; likelihood=:classification_multi)

    M = MLJDecisionTreeInterface.RandomForestClassifier()
    forest_model = MLJBase.machine(M, X, y)

    @test_throws ArgumentError RandomForestModel(forest_model; likelihood=:regression)
    @test_throws ArgumentError RandomForestModel(dummy; likelihood=:classification_binary)
    @test_throws ArgumentError RandomForestModel(dummy; likelihood=:classification_multi)
end
