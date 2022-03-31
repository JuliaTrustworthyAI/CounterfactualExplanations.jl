using CounterfactualExplanations
using CounterfactualExplanations.Models
using Random, LinearAlgebra
Random.seed!(1234)

@testset "Generic" begin

    w = [1.0 -2.0] # true coefficients
    b = [0]
    𝑴 = LogisticModel(w, b)
    x = [-1,0.5]
    p̅ = probs(𝑴, x)
    y = round(p̅[1])
    generator = GenericGenerator()

    @testset "Predetermined outputs" begin
        γ = 0.9
        target = round(probs(𝑴, x)[1])==0 ? 1 : 0 
        recourse = generate_counterfactual(generator, x, 𝑴, target, γ)
        @test recourse.target == target
        @test recourse.x == x
        @test recourse.y == y
        @test recourse.p̅ == p̅
    end

    @testset "Convergence" begin

        # Already in target and exceeding threshold probability:
        γ = probs(𝑴, x)[1]
        target = round(γ)
        recourse = generate_counterfactual(generator, x, 𝑴, target, γ)
        @test length(recourse.path)==1
        @test recourse.x == recourse.x̃
        @test recourse.y == recourse.ỹ
        @test recourse.p̅ == recourse.p̲
        @test recourse.converged == true

        # Threshold reached if converged:
        γ = 0.9
        target = round(probs(𝑴, x)[1])==0 ? 1 : 0 
        T = 1000
        recourse = generate_counterfactual(generator, x, 𝑴, target, γ; T=T)
        @test !recourse.converged || recourse.p̲[1] >= γ # either not converged or threshold reached
        @test !recourse.converged || recourse.ỹ >= recourse.y # either not covnerged or in target class
        @test !recourse.converged || length(recourse.path) <= T

    end

end

@testset "target_probs" begin

    using CounterfactualExplanations: target_probs

    @testset "Binary" begin
        p = [0.25]
        @test target_probs(p, 1) == [0.25]
        @test target_probs(p, 0) == [0.75]
        @test_throws DomainError target_probs(p, 2)
        @test_throws DomainError target_probs(p, -1)
    end

    @testset "Multi-class" begin
        p = [0.25, 0.75]
        @test target_probs(p, 1) == [0.25]
        @test target_probs(p, 2) == [0.75]
        @test_throws DomainError target_probs(p, 0)
        @test_throws DomainError target_probs(p, 1.1)
    end
end

@testset "threshold_reached" begin
    using CounterfactualExplanations: threshold_reached
    𝑴 = LogisticModel([1.0 -2.0], [0])
    x = [-1,0.5]
    p̅ = probs(𝑴, x)
    y = round(p̅[1])
    target = y == 1 ? 0 : 1
    ε = 1e-10
    
    @test threshold_reached(𝑴, x, y, 0.5+ε) == true
    @test threshold_reached(𝑴, x, target, 0.5+ε) == false

end

@testset "apply_mutability" begin
    using CounterfactualExplanations: apply_mutability
    𝑭 = [:both, :increase, :decrease, :none]
    @test apply_mutability([-1,1,-1,1], 𝑭)[4] == 0
    @test all(apply_mutability([-1,1,1,1], 𝑭)[[3,4]] .== 0)
    @test all(apply_mutability([-1,-1,-1,1], 𝑭)[[2,4]] .== 0)
    @test all(apply_mutability([-1,-1,1,1], 𝑭)[[2,3,4]] .== 0)
end

@testset "initialize_mutability" begin
    using CounterfactualExplanations: initialize_mutability
    struct SomeGenerator <: AbstractGenerator
        𝑭::Union{Nothing,Vector{Symbol}}
    end

    gen_unconstrained = SomeGenerator(nothing)
    gen_constrained = SomeGenerator([:none,:increase])

    @test length(initialize_mutability(gen_unconstrained, 1)) == 1
    @test length(initialize_mutability(gen_unconstrained, 2)) == 2
    @test all(initialize_mutability(gen_unconstrained, 2) .== :both)
    @test all(initialize_mutability(gen_constrained, 2) .== [:none,:increase])
    @test length(initialize_mutability(gen_constrained, 2)) == 2

end