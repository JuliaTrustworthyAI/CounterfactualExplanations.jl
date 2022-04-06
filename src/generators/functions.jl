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

An abstract type that serves as the base type for counterfactual generators. 
"""
abstract type AbstractGenerator end

# Loss:
using Flux
"""
    ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState)

    output = :logits

    loss = getfield(Losses, generator.loss)(
        getfield(Models, output)(counterfactual_state.M, counterfactual_state.x′), 
        counterfactual_state.target_encoded
    )    

    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
h(generator::AbstractGenerator, counterfactual_state::CounterfactualState) = generator.complexity(counterfactual_state.x-counterfactual_state.x′)


################################################################################
# --------------- Base type for gradient-based generator:
################################################################################
"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"""
    ∂ℓ(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the loss function at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂ℓ(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = gradient(() -> ℓ(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]

"""
    ∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the complexity penalty at the current counterfactual state for gradient-based generators. It assumes that `Zygote.jl` has gradient access.
"""
∂h(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = gradient(() -> h(generator, counterfactual_state), params(counterfactual_state.x′))[counterfactual_state.x′]

# Gradient:
"""
    ∇(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to compute the gradient of the counterfactual search objective for gradient-based generators. It simply computes the weighted sum over partial derivates. It assumes that `Zygote.jl` has gradient access.
"""
∇(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state) + generator.λ * ∂h(generator, counterfactual_state)

"""
    generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to generate feature perturbations for gradient-based generators through simple gradient descent.
"""
function generate_perturbations(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    Δx′ = - (generator.ϵ .* 𝐠ₜ) # gradient step
    return Δx′
end

"""
    mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to return mutability constraints that are dependent on the current counterfactual search state. For generic gradient-based generators, no state-dependent constraints are added.
"""
function mutability_constraints(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual_state.params[:mutability]
    return mutability # no additional constraints for GenericGenerator
end 

"""
    conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)

The default method to check if the all conditions for convergence of the counterfactual search have been satisified for gradient-based generators.
"""
function conditions_satisified(generator::AbstractGradientBasedGenerator, counterfactual_state::CounterfactualState)
    𝐠ₜ = ∇(generator, counterfactual_state)
    status = all(abs.(𝐠ₜ) .< generator.τ) 
    return status
end

# --------------- Specific generators:

# -------- Wachter et al (2018): 
struct GenericGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    complexity::Function # complexity function
    λ::AbstractFloat # strength of penalty
    ϵ::AbstractFloat # step size
    τ::AbstractFloat # tolerance for convergence
end

"""
    GenericGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        complexity::Function=norm,
        λ::AbstractFloat=0.1,
        ϵ::AbstractFloat=0.1,
        τ::AbstractFloat=1e-5
    )

An outer constructor method that instantiates a generic generator.

# Examples
```julia-repl
generator = GenericGenerator()
```
"""
GenericGenerator(
    ;
    loss::Symbol=:logitbinarycrossentropy,
    complexity::Function=norm,
    λ::AbstractFloat=0.1,
    ϵ::AbstractFloat=0.1,
    τ::AbstractFloat=1e-5
) = GenericGenerator(loss, complexity, λ, ϵ, τ)

# -------- Schut et al (2020): 
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Symbol # loss function
    δ::AbstractFloat # perturbation size
    n::Int # maximum number of times any feature can be changed
end

"""
    GreedyGenerator(
        ;
        loss::Symbol=:logitbinarycrossentropy,
        δ::Union{AbstractFloat,Nothing}=nothing,
        n::Union{Int,Nothing}=nothing
    )

An outer constructor method that instantiates a greedy generator.

# Examples

```julia-repl
generator = GreedyGenerator()
```
"""
function GreedyGenerator(
    ;
    loss::Symbol=:logitbinarycrossentropy,
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

    generator = GreedyGenerator(loss,δ,n)

    return generator
end

"""
    ∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState)    

he default method to compute the gradient of the counterfactual search objective for a greedy generator. Since no complexity penalty is needed, this gradients just correponds to the partial derivative with respect to the loss function.

"""
∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState) = ∂ℓ(generator, counterfactual_state)

"""
    generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState)

The default method to generate perturbations for a greedy generator. Only the most salient feature is perturbed.
"""
function generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState) 
    𝐠ₜ = ∇(generator, counterfactual_state) # gradient
    𝐠ₜ[counterfactual_state.params[:mutability] .== :none] .= 0
    Δx′ = reshape(zeros(length(counterfactual_state.x′)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δx′[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δx′
end

"""
    mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState)

The default method to return search state dependent mutability constraints for a greedy generator. Features that have been perturbed `n` times already can no longer be perturbed.
"""
function mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    mutability = counterfactual_state.params[:mutability]
    mutability[counterfactual_state.search[:times_changed_features] .>= generator.n] .= :none # constrains features that have already been exhausted
    return mutability
end 

"""
    conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState)

If all features have been perturbed `n` times already, then the search terminates.
"""
function conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState)
    status = all(counterfactual_state.search[:times_changed_features].>=generator.n)
    return status
end