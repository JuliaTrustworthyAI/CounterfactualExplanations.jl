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
function hinge_loss_ir(
    ce::AbstractCounterfactualExplanation
)
    return max(0, invalidation_rate(ce) - ce.convergence.invalidation_rate)
end

"""
    invalidation_rate(ce::AbstractCounterfactualExplanation)

Calculate the invalidation rate of a counterfactual explanation.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the invalidation rate for.
- `kwargs`: Additional keyword arguments to pass to the function.

# Returns
The invalidation rate of the counterfactual explanation.
"""
function invalidation_rate(ce::AbstractCounterfactualExplanation)
    index_target = findfirst(map(x -> x == ce.target, ce.data.y_levels))
    f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[index_target]
    grad = []
    for i in 1:length(ce.s′)
        push!(
            grad,
            Flux.gradient(
                () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[i],
                Flux.params(ce.s′),
            )[ce.s′],
        )
    end
    gradᵀ = LinearAlgebra.transpose(grad)

    identity_matrix = LinearAlgebra.Matrix{Float32}(I, length(grad), length(grad))
    denominator = sqrt(gradᵀ * ce.convergence.variance * identity_matrix * grad)[1]

    normalized_gradient = f_loss / denominator
    ϕ = Distributions.cdf(Distributions.Normal(0, 1), normalized_gradient)
    return 1 - ϕ
end
