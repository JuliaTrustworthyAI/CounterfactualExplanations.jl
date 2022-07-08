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
    ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

    loss_fun = !isnothing(generator.loss) ? getfield(Losses, generator.loss) : CounterfactualState.guess_loss(counterfactual_state)
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(
        getfield(Models, :logits)(counterfactual_state.M, counterfactual_state.f(counterfactual_state.s′)), 
        counterfactual_state.target_encoded
    )    

    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
h(generator::AbstractGenerator, counterfactual_state::CounterfactualState.State) = generator.complexity(
    counterfactual_state.x .- counterfactual_state.f(counterfactual_state.s′)
)

################################################################################
# Subtypes
################################################################################

include("gradient_based/functions.jl")