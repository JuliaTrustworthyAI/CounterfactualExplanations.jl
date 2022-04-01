################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct CounterfactualState
    x::AbstractArray
    target_encoded::Union{Number, AbstractVector}
    x′::AbstractArray
    M::AbstractFittedModel
    params::Dict
    search::Union{Dict,Nothing}
end

################################################################################
# --------------- Base type for generator:
################################################################################
"""
    AbstractGenerator

An abstract type that serves as the base type for recourse generators. 
"""
abstract type AbstractGenerator end

# Loss:
using Flux
function ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState)

    output = :logits

    loss = getfield(Losses, generator.loss)(
        getfield(Models, output)(counterfactual_state.M, counterfactual_state.x′), 
        counterfactual_state.target_encoded
    )    

    return loss
end

# Complexity:
h(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = generator.complexity(counterfactual_state.x-counterfactual_state.x′)


################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

∂ℓ(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = gradient(() -> ℓ(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]

∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = gradient(() -> h(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]

# Gradient:
∇(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state) + generator.λ * ∂h(generator, counterfactual_state)

function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    Δx′ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx′
end

function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual_state.params[:mutability]
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
- [`generate_counterfactual(generator::AbstractGradientBasedGenerator, x::Vector, M::Models.AbstractFittedModel, target::AbstractFloat; T=1000)`](@ref)
"""
struct GenericGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    complexity::Function # complexity function
    mutability::Union{Nothing,Vector{Symbol}} # mutibility constraints 
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # step size
    τ::AbstractFloat # tolerance for convergence
end

GenericGenerator(
    ;
    loss::Symbol=:logitbinarycrossentropy,
    complexity::Function=norm,
    mutability::Union{Nothing,Vector{Symbol}}=nothing,
    λ::AbstractFloat=0.1,
    ϵ::AbstractFloat=0.1,
    τ::AbstractFloat=1e-5
) = GenericGenerator(loss, complexity, mutability, λ, ϵ, τ)

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
- [`generate_counterfactual(generator::AbstractGradientBasedGenerator, x::Vector, M::Models.AbstractFittedModel, target::AbstractFloat; T=1000)`](@ref)
"""
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    mutability::Union{Nothing,Vector{Symbol}} # mutibility constraints 
    δ::AbstractFloat # perturbation size
    n::Int # maximum number of times any feature can be changed
end

function GreedyGenerator(
    ;
    loss::Symbol=:logitbinarycrossentropy,
    mutability::Union{Nothing,Vector{Symbol}}=nothing,
    δ::Union{AbstractFloat,Nothing}=nothing,
    n::Union{Int,Nothing}=nothing
) 
    if all(isnothing.([δ, n])) 
        δ = 0.1
        n = 10
    elseif isnothing(δ) && !isnothing(n)
        δ = 1/n
    elseif !isnothing(δ) && isnothing(n)
        n = 1/δ
    end

    generator = GreedyGenerator(loss,mutability,δ,n)

    return generator
end


∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state)

function generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    𝐠ₜ[counterfactual_state.params[:mutability] .== :none] .= 0
    Δx′ = reshape(zeros(length(counterfactual_state.x′)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δx′[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δx′
end

function mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual_state.params[:mutability]
    mutability[counterfactual_state.search[:times_changed_features] .>= generator.n] .= :none # constrains features that have already been exhausted
    return mutability
end 

function conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    status = all(counterfactual_state.search[:times_changed_features].>=generator.n)
    return status
end