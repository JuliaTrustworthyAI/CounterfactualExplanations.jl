using ChainRulesCore: ignore_derivatives
using Distributions: Distributions
import DifferentiationInterface as DI
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
function invalidation_rate(ce_state::AbstractArray, ce::AbstractCounterfactualExplanation)
    index_target = get_target_index(ce.data.y_levels, ce.target)
    f_loss = logits(ce.M, CounterfactualExplanations.decode_state(ce))[index_target]
    y = ce.target_encoded

    # Create closure
    function f(x)
        return logits(ce.M, CounterfactualExplanations.decode_state(ce, x))[index_target]
    end

    # Get AD backend:
    backend = get!(ce.search, :ad_backend, CounterfactualExplanations.choose_ad_backend(ce))

    # Get prepared gradient:
    prep = get!(ce.search, :prep_inval, DI.prepare_gradient(f, backend, similar(ce_state)))

    # Compute gradient:
    grad = DI.gradient(f, prep, backend)
    denominator = sqrt(ce.convergence.variance) * norm(grad)
    normalized_gradient = f_loss / denominator
    ϕ = Distributions.cdf(Distributions.Normal(0, 1), normalized_gradient)
    return 1 - ϕ
end

"""
    invalidation_rate(ce::AbstractCounterfactualExplanation)

Single-argument method for convenience.
"""
function invalidation_rate(ce::AbstractCounterfactualExplanation)
    cf = CounterfactualExplanations.decode_state(ce)
    return invalidation_rate(cf, ce)
end
