using CLEAR
using CLEAR.Losses
using Random

@testset "Hinge" begin

    # Equality
    t = rand([-1,1]) # target
    a = t # logits
    @test hinge_loss(a,t) == 0.0

    # Label mapping
    t = rand([0,1])
    a = t==0 ? -1 : 1
    @test hinge_loss(a,t) == 0.0

    @test hinge_loss(-1,1) == 2.0
    @test hinge_loss(100,1) == 0.0
end

@testset "Flux functions" begin
    @test !isnothing(logitbinarycrossentropy)
    @test !isnothing(logitcrossentropy)
    @test !isnothing(mse)
    @test !isnothing(mae)
end