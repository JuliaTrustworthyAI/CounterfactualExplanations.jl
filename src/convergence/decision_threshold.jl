struct DecisionThresholdConvergence <: AbstractConvergence
    decision_threshold::AbstractFloat
    max_iter::Int
    min_success_rate::AbstractFloat
end

function DecisionThresholdConvergence(;
    decision_threshold::AbstractFloat=0.5,
    max_iter::Int=100,
    min_success_rate::AbstractFloat=0.75,
    y_levels::Union{Nothing,AbstractVector}=nothing,
)
    @assert 0.0 < min_success_rate <= 1.0 "Minimum success rate should be ∈ [0.0,1.0]."
    if isa(y_levels, AbstractVector)
        decision_threshold = 1 / length(y_levels)
    end
    return DecisionThresholdConvergence(decision_threshold, max_iter, min_success_rate)
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
