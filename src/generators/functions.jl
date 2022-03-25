################################################################################
# --------------- Base type for generator:
################################################################################
"""
    AbstractGenerator

An abstract type that serves as the base type for recourse generators. 
"""
abstract type AbstractGenerator end
struct CounterfactualState
    x̅::AbstractArray
    target::Number
    x̲::AbstractArray
    𝑴::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

# Loss:
ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = getfield(Losses, generator.loss)(
    Models.logits(counterfactual_state.𝑴, counterfactual_state.x̲), counterfactual_state.target
)
∂ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = gradient(() -> ℓ(generator, counterfactual_state), params(x̲))[x̲]

# Complexity:
h(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = generator.complexity(counterfactual_state.x̅-counterfactual_state.x̲)
∂h(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = gradient(() -> h(generator, counterfactual_state), params(x̲))[x̲]

################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

# Gradient:
∇(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state) + generator.λ * ∂h(generator, counterfactual_state)

function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    Δx̲ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx̲
end

function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    𝐠ₜ = ∇(generator, counterfactual_state)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
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
- [`generate_counterfactual(generator::AbstractGenerator, x̅::Vector, 𝑴::Models.AbstractFittedModel, target::AbstractFloat; T=1000)`](@ref)
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
- [`generate_counterfactual(generator::AbstractGenerator, x̅::Vector, 𝑴::Models.AbstractFittedModel, target::AbstractFloat; T=1000)`](@ref)
"""
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    𝑭::Union{Nothing,Vector{Symbol}} # mutibility constraints 
    δ::AbstractFloat # perturbation size
    n::Int # maximum number of times any feature can be changed
end

GreedyGenerator() = GreedyGenerator(:logitbinarycrossentropy,nothing,0.1,10)

∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state)

function generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    𝐠ₜ[counterfactual.params[:mutability] .== :none] .= 0
    Δx̲ = reshape(zeros(length(x̲)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δx̲[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δx̲
end

function mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual.params[:mutability]
    mutability[counterfactual.search[:times_changed_features] .>= generator.n] .= :none # constrains features that have already been exhausted
    return mutability
end 

function conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    status = all(counterfactual.search[:times_changed_features].>=generator.n)
    return status
end