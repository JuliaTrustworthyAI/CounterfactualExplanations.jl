# -------- Schut et al (2020): 
struct GreedyGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Symbol} # loss function
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
    loss::Union{Nothing,Symbol}=nothing,
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

# API streamlining:
using Parameters
@with_kw struct GreedyGeneratorParams
    δ::AbstractFloat=nothing
    n::AbstractFloat=nothing
end

GreedyGenerator(
    ;
    loss::Union{Nothing,Symbol}=nothing,
    complexity::Nothing=nothing,
    λ::Nothing=nothing,
    params::GreedyGeneratorParams=GreedyGeneratorParams()
) = GreedyGenerator(loss=loss, δ=params.δ, n=params.n)

"""
    ∇(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)    

he default method to compute the gradient of the counterfactual search objective for a greedy generator. Since no complexity penalty is needed, this gradients just correponds to the partial derivative with respect to the loss function.

"""
∇(generator::GreedyGenerator, M::Models.Models.AbstractDifferentiableJuliaModel, counterfactual_state::CounterfactualState.State) = ∂ℓ(generator, M, counterfactual_state)

"""
    generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

The default method to generate perturbations for a greedy generator. Only the most salient feature is perturbed.
"""
function generate_perturbations(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State) 
    𝐠ₜ = ∇(generator, counterfactual_state.M, counterfactual_state) # gradient
    𝐠ₜ[counterfactual_state.params[:mutability] .== :none] .= 0
    Δs′ = reshape(zeros(length(counterfactual_state.s′)), size(𝐠ₜ))
    iₜ = argmax(abs.(𝐠ₜ)) # choose most salient feature
    Δs′[iₜ] -= generator.δ * sign(𝐠ₜ[iₜ]) # counterfactual update
    return Δs′
end

"""
    mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

The default method to return search state dependent mutability constraints for a greedy generator. Features that have been perturbed `n` times already can no longer be perturbed.
"""
function mutability_constraints(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)
    mutability = counterfactual_state.params[:mutability]
    mutability[counterfactual_state.search[:times_changed_features] .>= generator.n] .= :none # constrains features that have already been exhausted
    return mutability
end 

"""
    conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)

If all features have been perturbed `n` times already, then the search terminates.
"""
function conditions_satisified(generator::GreedyGenerator, counterfactual_state::CounterfactualState.State)
    status = all(counterfactual_state.search[:times_changed_features].>=generator.n)
    return status
end