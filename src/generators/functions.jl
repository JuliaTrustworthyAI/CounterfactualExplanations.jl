################################################################################
# --------------- Base type for generator:
################################################################################
"""
    AbstractGenerator

An abstract type that serves as the base type for recourse generators. 
"""
abstract type AbstractGenerator end

# Loss:
ℓ(generator::AbstractGenerator, x̲, 𝑴, t) = getfield(Losses, generator.loss)(Models.logits(𝑴, x̲), t)
∂ℓ(generator::AbstractGenerator, x̲, 𝑴, t) = gradient(() -> ℓ(generator, x̲, 𝑴, t), params(x̲))[x̲]

# Complexity:
h(generator::AbstractGenerator, x̅, x̲) = generator.complexity(x̅-x̲)
∂h(generator::AbstractGenerator, x̅, x̲) = gradient(() -> h(generator::AbstractGenerator, x̅, x̲), params(x̲))[x̲]


abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

# Gradient:
∇(generator::AbstractGradientBasedGenerator, x̲, 𝑴, t, x̅) = ∂ℓ(generator, x̲, 𝑴, t) + generator.λ * ∂h(generator::AbstractGradientBasedGenerator, x̅, x̲)

function generate_perturbations(generator::AbstractGradientBasedGenerator, x̲, 𝑴, t, x̅, 𝑭ₜ) 
    𝐠ₜ = ∇(generator, x̲, 𝑴, t, x̅) # gradient
    Δx̲ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx̲
end

function mutability_constraints(generator::AbstractGradientBasedGenerator, 𝑭ₜ, 𝑷)
    return 𝑭ₜ # no additional constraints for GenericGenerator
end 

function conditions_satisified(generator::AbstractGradientBasedGenerator, x̲, 𝑴, t, x̅, 𝑷)
    𝐠ₜ = ∇(generator, x̲, 𝑴, t, x̅)
    all(abs.(𝐠ₜ) .< generator.τ) 
end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
"""
    GenericGenerator(λ::AbstractFloat, ϵ::AbstractFloat, τ::AbstractFloat, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})

A constructor for a generic recourse generator. It takes values for the complexity penalty `λ`, the learning rate `ϵ`, the tolerance for convergence `τ`, 
    the type of `loss` function to be used in the recourse objective and a mutability constraint mask `𝑭`.

# Examples
```julia-repl
generator = GenericGenerator(0.1,0.1,1e-5,:logitbinarycrossentropy,nothing)
```

See also:
- [`generate_counterfactual(generator::AbstractGenerator, x̅::Vector, 𝑴::Models.FittedModel, target::AbstractFloat; T=1000)`](@ref)
"""
struct GenericGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    complexity::Function # complexity function
    𝑭::Union{Nothing,Vector{Symbol}} # mutibility constraints 
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # step size
    τ::AbstractFloat # tolerance for convergence
end

GenericGenerator() = GenericGenerator(:logitbinarycrossentropy,norm,nothing,0.1,0.1,1e-5)

################################################################################
# -------- Schut et al (2021):
################################################################################
"""
    GreedyGenerator(δ::AbstractFloat, n::Int, loss::Symbol, 𝑭::Union{Nothing,Vector{Symbol}})

Constructs a greedy recourse generator for Bayesian models. It takes values for the perturbation size `δ`, the maximum number of times `n` that any feature can be changed, 
    the type of `loss` function to be used in the recourse objective and a mutability constraint mask `𝑭`.

# Examples
```julia-repl
generator = GreedyGenerator(0.01,20,:logitbinarycrossentropy, nothing)
```

See also:
- [`generate_counterfactual(generator::AbstractGenerator, x̅::Vector, 𝑴::Models.FittedModel, target::AbstractFloat; T=1000)`](@ref)
"""
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    𝑭::Union{Nothing,Vector{Symbol}} # mutibility constraints 
    δ::AbstractFloat # perturbation size
    n::Int # maximum number of times any feature can be changed
end

GreedyGenerator() = GreedyGenerator(:logitbinarycrossentropy,nothing,0.1,10)

∇(generator::GreedyGenerator, x̲, 𝑴, t, x̅) = ∂ℓ(generator, x̲, 𝑴, t)

function generate_perturbations(generator::GreedyGenerator, x̲, 𝑴, t, x̅, 𝑭ₜ) 
    𝐠ₜ = ∇(generator, x̲, 𝑴, t, x̅) # gradient
    𝐠ₜ[𝑭ₜ .== :none] .= 0
    Δx̲ = reshape(zeros(length(x̲)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δx̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δx̲
end

function mutability_constraints(generator::GreedyGenerator, 𝑭ₜ, 𝑷)
    𝑭ₜ[𝑷 .>= generator.n] .= :none # constraints features that have already been exhausted
    return 𝑭ₜ
end 

function conditions_satisified(generator::GreedyGenerator, x̲, 𝑴, t, x̅, 𝑷)
    feature_changes_exhausted = all(𝑷.>=generator.n)
    return feature_changes_exhausted 
end