using CounterfactualExplanations.Models: build_mlp
using Flux: Chain

@testset "MLP" begin
    build_mlp() |> x -> @test typeof(x) <: Chain
    build_mlp(; dropout=true) |> x -> @test typeof(x) <: Chain
    build_mlp(; batch_norm=true) |> x -> @test typeof(x) <: Chain
end
