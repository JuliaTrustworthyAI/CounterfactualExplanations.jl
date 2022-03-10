using CLEAR
using CLEAR.Losses
using Random

@testset "Hinge" begin
    a = rand([-1,1])
    @test hinge_loss(a,a) == 0.0

    a = rand([0,1])
    @test hinge_loss(a,a) == 0.0

    @test hinge_loss(-1,1) == 2.0
    @test hinge_loss(100,1) == 0.0
end

@testset "Flux functions" begin
    @test !isnothing(logitbinarycrossentropy)
    @test !isnothing(logitcrossentropy)
    @test !isnothing(mse)
    @test !isnothing(mae)
end