Base.@kwdef struct InvalidationRateConvergence <: AbstractConvergence
    invalidation_rate::AbstractFloat = 0.1
    max_iter::Int = 100
    variance::AbstractFloat = 0.01
end

"""
    converged(convergence::InvalidationRateConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is invalidation rate.
"""
function converged(
    convergence::InvalidationRateConvergence, ce::AbstractCounterfactualExplanation
)
    ir = Objectives.invalidation_rate(ce)
    label = Models.predict_label(ce.M, ce.data, ce.x′)[1]
    return label == ce.target && convergence.invalidation_rate > ir
end
