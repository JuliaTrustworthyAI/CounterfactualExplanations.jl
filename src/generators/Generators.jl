# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..Models
using Flux
using LinearAlgebra

export Generator

# --------------- Base type for generator:
abstract type Generator end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
struct GenericGenerator <: Generator
    λ::Float64 # strength of penalty
    ϵ::Float64 # step size
    τ::Float64 # tolerance for convergence
end

ℓ(generator::GenericGenerator, x, 𝓜, t) = Flux.Losses.logitbinarycrossentropy(Models.logits(𝓜, x), t)
complexity(generator::GenericGenerator, x̅, x̲) = norm(x̅-x̲)
objective(generator::GenericGenerator, x̲, 𝓜, t, x̅) = ℓ(generator, x̲, 𝓜, t) + generator.λ * complexity(generator, x̅, x̲) 
∇(generator::GenericGenerator, x̲, 𝓜, t, x̅) = gradient(() -> objective(generator, x̲, 𝓜, t, x̅), params(x̲))[x̲]

function step(generator::GenericGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    return x̲ - (generator.ϵ .* 𝐠ₜ)
end

function convergence(generator::GenericGenerator, x̲, 𝓜, t, x̅)
    𝐠ₜ = ∇(generator, x̲, 𝓜, t, x̅)
    all(abs.(𝐠ₜ) .< generator.τ)
end

# -------- Schut et al (2021):
struct GreedyGenerator <: Generator
    Γ::Float64 # desired level of confidence 
    δ::Float64 # perturbation size
    n::Int64 # maximum number of times any feature can be changed
end

ℓ(generator::GreedyGenerator, x, 𝓜, t) = - (t * log(𝛔(𝓜(x))) + (1-t) * log(1-𝛔(𝓜(x))))
objective(generator::GreedyGenerator, x̲, 𝓜, t) = ℓ(generator, x̲, 𝓜, t) 
∇(generator::GreedyGenerator, x̲, 𝓜, t) = gradient(() -> objective(generator, x̲, 𝓜, t), params(x̲))

function step(generator::GreedyGenerator, x̲, 𝓜, t, x̅, 𝓘) 
    𝐠ₜ = ∇(generator, x̲, 𝓜, t)
    𝐠ₜ[𝓘] .= 0 # set gradient of immutable features to zero
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    x̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return x̲
end

function convergence(generator::GreedyGenerator, x̲, 𝓜, t, x̅)
    𝓜.confidence(x̲) .> generator.Γ
end

end