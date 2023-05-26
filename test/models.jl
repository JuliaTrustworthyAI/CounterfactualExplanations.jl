using CounterfactualExplanations
using CounterfactualExplanations.Models
using Flux
using LinearAlgebra
using MLUtils
using Random
using MLJ

@testset "Models for synthetic data" begin
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


@testset "MLJ models for synthetic data" begin
    evotree = @load EvoTreeClassifier pkg=EvoTrees
    nn = @load NeuralNetworkClassifier pkg=MLJFlux
    xgboost = @load XGBoostClassifier pkg=XGBoost
    mlj_models = [evotree, nn, xgboost]
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X
            for model in mlj_models
                name = string(model)
                df = DataFrame(X', :auto)
                mach = machine(model, df, categorical(y)) |> fit!
                M = CounterfactualExplanations.Models.MLJModel(mach)
                @testset "$name" begin
                    @testset "Matrix of inputs" begin
                        @test size(logits(M, X))[2] == size(X, 2)
                        @test size(probs(M, X))[2] == size(X, 2)
                    end
                    @testset "Vector of inputs" begin
                        @test size(logits(M, X[:, 1]), 2) == 1
                        @test size(probs(M, X[:, 1]), 2) == 1
                    end
                end
            end
        end
    end
end