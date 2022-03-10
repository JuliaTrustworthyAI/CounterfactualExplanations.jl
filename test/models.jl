using CLEAR
using CLEAR.Models
using Random
using LinearAlgebra

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
end