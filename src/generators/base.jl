################################################################################
# --------------- Constructor for counterfactual state:
################################################################################
struct CounterfactualState
    x::AbstractArray
    target_encoded::Union{Number, AbstractVector}
    x′::AbstractArray
    z′::Union{AbstractArray, Nothing}
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

    output = :logits # currently counterfactual loss is always computed with respect to logits

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
# Subtypes
################################################################################

include("gradient_based/functions.jl")