using CounterfactualExplanations
using CounterfactualExplanations.Models
using DataFrames
using Flux
using LinearAlgebra
using MLJ
using MLUtils
using Random

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
    evotree = @load EvoTreeClassifier pkg = EvoTrees
    mlj_models = [evotree]
    for (key, value) in synthetic
        name = string(key)
        @testset "$name" begin
            X = value[:data].X
            y = value[:data].y
            println(size(X))
            println(size(y))
            println("@@@@@@@@@@@")
            println(y)
            for model in mlj_models
                name = string(model)
                @testset "$name" begin
                    @testset "Matrix of inputs" begin
                        df = DataFrame(X', :auto)
                        mach = machine(model(), df, categorical(y[:, 1]')) |> fit!
                        M = CounterfactualExplanations.Models.MLJModel(mach)
                        @test size(logits(M, X))[2] == size(X, 2)
                        @test size(probs(M, X))[2] == size(X, 2)
                    end
                    @testset "Vector of inputs" begin
                        df = DataFrame(X[:, 1]', :auto)
                        mach = machine(model(), df, categorical(y')) |> fit!
                        M = CounterfactualExplanations.Models.MLJModel(mach)
                        @test size(logits(M, X[:, 1]), 2) == 1
                        @test size(probs(M, X[:, 1]), 2) == 1
                    end
                end
            end
        end
    end
end