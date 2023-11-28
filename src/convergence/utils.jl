"""
    threshold_reached(ce::CounterfactualExplanation)

Determines if the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(ce::CounterfactualExplanation)
    γ = ce.convergence.decision_threshold
    success_rate = sum(target_probs(ce) .>= γ) / ce.num_counterfactuals
    return success_rate > ce.convergence.min_success_rate
end

"""
    get_convergence_type(convergence::AbstractConvergence)

Returns the convergence object.
"""
function get_convergence_type(convergence::AbstractConvergence)
    return convergence
end

"""
    get_convergence_type(convergence::Symbol)

Returns the convergence object from the dictionary of default convergence types.
"""
function get_convergence_type(convergence::Symbol)
    return get(
        convergence_catalogue,
        convergence,
        error("Convergence criterion not recognized: $convergence."),
    )
end
