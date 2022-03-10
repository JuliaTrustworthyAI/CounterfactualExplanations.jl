using CLEAR
using Test
using Random
Random.seed!(0)

@testset "CLEAR.jl" begin

    @testset "Utils" begin
        include("utils.jl")
    end

    @testset "Losses" begin
        include("losses.jl")
    end

    @testset "Model" begin
        include("models.jl")
    end

end
