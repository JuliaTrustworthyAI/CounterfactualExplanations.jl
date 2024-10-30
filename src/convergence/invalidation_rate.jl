using ChainRulesCore: ignore_derivatives
using Distributions: Distributions
using Flux: Flux
using LinearAlgebra: LinearAlgebra

Base.@kwdef struct InvalidationRateConvergence <: AbstractConvergence
    invalidation_rate::AbstractFloat = 0.1
    max_iter::Int = 100
    variance::AbstractFloat = 0.01
end

"""
    converged(
        convergence::InvalidationRateConvergence,
        ce::AbstractCounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    )

Checks if the counterfactual search has converged when the convergence criterion is invalidation rate.
"""
function converged(
    convergence::InvalidationRateConvergence,
    ce::AbstractCounterfactualExplanation,
    x::Union{AbstractArray,Nothing}=nothing,
)
    ir = invalidation_rate(ce)
    label = Models.predict_label(ce.M, ce.data, ce.counterfactual)[1]
    return label == ce.target && convergence.invalidation_rate > ir
end

"""
    invalidation_rate(ce::AbstractCounterfactualExplanation)

Calculates the invalidation rate of a counterfactual explanation.

# Arguments
- `ce::AbstractCounterfactualExplanation`: The counterfactual explanation to calculate the invalidation rate for.
- `kwargs`: Additional keyword arguments to pass to the function.

# Returns
The invalidation rate of the counterfactual explanation.
"""
function invalidation_rate(ce::AbstractCounterfactualExplanation)
    z = []
    ignore_derivatives() do
        index_target = get_target_index(ce.data.y_levels, ce.target)
        f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[index_target]
        grad = Flux.gradient(
            () -> logits(ce.M, CounterfactualExplanations.decode_state(ce))[index_target],
            Flux.params(ce.counterfactual_state),
        )[ce.counterfactual_state]
        gradᵀ = LinearAlgebra.transpose(grad)
        denominator = sqrt(gradᵀ * UniformScaling(ce.convergence.variance) * grad)[1]
        normalized_gradient = f_loss / denominator
        push!(z, normalized_gradient)
    end
    ϕ = Distributions.cdf(Distributions.Normal(0, 1), z[1])
    return 1 - ϕ
end
