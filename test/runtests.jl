using CLEAR
using Test
using Random
Random.seed!(0)

@testset "CLEAR.jl" begin

    @testset "Utils" begin
        include("utils.jl")
    end

end
