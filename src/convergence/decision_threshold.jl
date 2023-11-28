Base.@kwdef struct DecisionThresholdConvergence <: AbstractConvergence
    max_iter::Int = 100
    decision_threshold::AbstractFloat = 0.5
    min_success_rate::AbstractFloat = 0.75
end

"""
    converged(convergence::DecisionThresholdConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is the decision threshold.
"""
function converged(
    convergence::DecisionThresholdConvergence, ce::AbstractCounterfactualExplanation
)
    return threshold_reached(ce)
end

"""
    threshold_reached(ce::CounterfactualExplanation)

Determines if the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(ce::AbstractCounterfactualExplanation)
    γ = ce.convergence.decision_threshold
    success_rate = sum(target_probs(ce) .>= γ) / ce.num_counterfactuals
    return success_rate > ce.convergence.min_success_rate
end
