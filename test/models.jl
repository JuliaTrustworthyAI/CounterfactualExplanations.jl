using CounterfactualExplanations
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLJ
using MLUtils
using PythonCall
using Random

include("pytorch.jl")

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
                    end
                    @testset "Vector of inputs" begin
                        @test size(logits(model[:model], X[:, 1]), 2) == 1
                        @test size(probs(model[:model], X[:, 1]), 2) == 1
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
