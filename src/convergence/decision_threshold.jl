Base.@kwdef struct DecisionThresholdConvergence <: AbstractConvergence
    max_iter::Int = 100
    decision_threshold::AbstractFloat = 0.5
    min_success_rate::AbstractFloat = 0.75
end

"""
    converged(convergence::DecisionThresholdConvergence, ce::CounterfactualExplanation)

Checks if the counterfactual search has converged when the convergence criterion is the decision threshold.
"""
function converged(convergence::DecisionThresholdConvergence, ce::CounterfactualExplanation)
    return threshold_reached(ce)
end
