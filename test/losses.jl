using CounterfactualExplanations
using CounterfactualExplanations.Losses
using Random

@testset "Flux functions" begin
    @test !isnothing(logitbinarycrossentropy)
    @test !isnothing(logitcrossentropy)
    @test !isnothing(mse)
    @test !isnothing(mae)
end