# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..Models
using Flux
using LinearAlgebra

export Generator, GenericGenerator

"""
    Generator

An abstract type that serves as the base type for recourse generators. 
"""

# --------------- Base type for generator:
abstract type Generator end

# --------------- Specific generators:

# -------- Wachter et al (2018): 

"""
    GenericGenerator(λ::Float64, ϵ::Float64, τ::Float64)

A constructor for a generic recourse generator. It takes values for the complexity penalty `λ`, the learning rate `ϵ` and the tolerance for convergence `τ`.

# Examples
```julia-repl
generator = GenericGenerator(0.1,0.1,1e-5)
```

See also [`generate_recourse`](@ref)
"""
struct GenericGenerator <: Generator
    λ::Float64 # strength of penalty
    ϵ::Float64 # step size
    τ::Float64 # tolerance for convergence
end

ℓ(generator::GenericGenerator, x, 𝑴, t) = Flux.Losses.logitbinarycrossentropy(Models.logits(𝑴, x), t)
complexity(generator::GenericGenerator, x̅, x̲) = norm(x̅-x̲)
objective(generator::GenericGenerator, x̲, 𝑴, t, x̅) = ℓ(generator, x̲, 𝑴, t) + generator.λ * complexity(generator, x̅, x̲) 
∇(generator::GenericGenerator, x̲, 𝑴, t, x̅) = gradient(() -> objective(generator, x̲, 𝑴, t, x̅), params(x̲))[x̲]

function step(generator::GenericGenerator, x̲, 𝑴, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝑴, t, x̅)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    return x̲ - (generator.ϵ .* 𝐠ₜ)
end

function convergence(generator::GenericGenerator, x̲, 𝑴, t, x̅)
    𝐠ₜ = ∇(generator, x̲, 𝑴, t, x̅)
    all(abs.(𝐠ₜ) .< generator.τ)
end

# -------- Schut et al (2021):
struct GreedyGenerator <: Generator
    Γ::Float64 # desired level of confidence 
    δ::Float64 # perturbation size
    n::Int64 # maximum number of times any feature can be changed
end

ℓ(generator::GreedyGenerator, x, 𝑴, t) = - (t * log(𝛔(𝑴(x))) + (1-t) * log(1-𝛔(𝑴(x))))
objective(generator::GreedyGenerator, x̲, 𝑴, t) = ℓ(generator, x̲, 𝑴, t) 
∇(generator::GreedyGenerator, x̲, 𝑴, t) = gradient(() -> objective(generator, x̲, 𝑴, t), params(x̲))

function step(generator::GreedyGenerator, x̲, 𝑴, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝑴, t)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    x̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return x̲
end

function convergence(generator::GreedyGenerator, x̲, 𝑴, t, x̅)
    𝑴.confidence(x̲) .> generator.Γ
end

end