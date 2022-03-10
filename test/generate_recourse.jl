using CLEAR
using CLEAR.Models
using Random, LinearAlgebra
Random.seed!(1234)

@testset "Generic" begin

    w = [1.0 -2.0] # true coefficients
    b = [0]
    𝑴 = LogisticModel(w, b)
    x̅ = [-1,0.5]
    p̅ = probs(𝑴, x̅)
    y̅ = round(p̅[1])
    generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)

    @testset "Predetermined outputs" begin
        γ = 0.9
        target = round(probs(𝑴, x̅)[1])==0 ? 1 : 0 
        recourse = generate_recourse(generator, x̅, 𝑴, target, γ)
        @test recourse.target == target
        @test recourse.x̅ == x̅
        @test recourse.y̅ == y̅
        @test recourse.p̅ == p̅
    end

    @testset "Convergence" begin

        # Already in target and exceeding threshold probability:
        γ = probs(𝑴, x̅)[1]
        target = round(γ)
        recourse = generate_recourse(generator, x̅, 𝑴, target, γ)
        @test length(recourse.path)==1
        @test recourse.x̅ == recourse.x̲
        @test recourse.y̅ == recourse.y̲
        @test recourse.p̅ == recourse.p̲
        @test recourse.converged == true

        # Threshold reached if converged:
        γ = 0.9
        target = round(probs(𝑴, x̅)[1])==0 ? 1 : 0 
        T = 1000
        recourse = generate_recourse(generator, x̅, 𝑴, target, γ; T=T)
        @test !recourse.converged || recourse.p̲[1] >= γ # either not converged or threshold reached
        @test !recourse.converged || recourse.y̲ >= recourse.y̅ # either not covnerged or in target class
        @test !recourse.converged || length(recourse.path) <= T

    end

end