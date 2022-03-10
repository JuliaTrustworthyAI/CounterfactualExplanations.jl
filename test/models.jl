using CLEAR
using CLEAR.Models
using Random
using LinearAlgebra
using NNlib

@testset "Exceptions" begin
    @testset "LogisticModel" begin
        w = [1,2] # vector instead of matrix
        b = 0 # scalar instead of array
        @test_throws MethodError LogisticModel(w, b)
    end

    @testset "BayesianLogisticModel" begin

        μ = [0, 1.0, -2.0] # vector instead of matrix
        Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) # MAP covariance matrix
        @test_throws MethodError BayesianLogisticModel(μ, Σ)

        # Dimensions not matching:
        μ = [0 1.0] 
        Σ = Symmetric(reshape(randn(9),3,3).*0.1 + UniformScaling(1.0)) 
        @test_throws DimensionMismatch BayesianLogisticModel(μ, Σ)

    end

    @testset "logits" begin
        𝑴(x) = [2 1] * x # model not declared as subtype of FittedModel
        x = [1,1]
        @test_throws MethodError logits(𝑴, x)
    end

    @testset "probs" begin
        𝑴(x) = [2 1] * x # model not declared as subtype of FittedModel
        x = [1,1]
        @test_throws MethodError probs(𝑴, x)
    end
end

@testset "Predictions" begin

    𝑴 = LogisticModel([1 1],[0])
    x = [1,1]
    @test logits(𝑴, x)[1] == 2
    @test probs(𝑴, x)[1] == σ(2)

end