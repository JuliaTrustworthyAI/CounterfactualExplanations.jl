using Distributions
using LinearAlgebra

"""
	Flux.Losses.logitbinarycrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitbinarycrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitbinarycrossentropy(
    ce::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.logitbinarycrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitcrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.logitcrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.mse(ce::AbstractCounterfactualExplanation)

Simply extends the `mse` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.mse(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.mse(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
    hinge_loss_ir(convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation)

Calculate the hinge loss of a counterfactual explanation with respect to the probability of invalidation following: https://openreview.net/forum?id=sC-PmTsiTB.

# Arguments
- `convergence::InvalidationRateConvergence`: The convergence criterion to use.
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the hinge loss for.

# Returns
The hinge loss of the counterfactual explanation.
"""
function hinge_loss_ir(ce::AbstractCounterfactualExplanation)
    return max(0, invalidation_rate(ce) - ce.convergence.invalidation_rate)
end
