"""
    terminated(ce::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has terminated.
"""
function terminated(ce::CounterfactualExplanation)
    return converged(ce) || steps_exhausted(ce)
end

"""
    in_target_class(ce::CounterfactualExplanation)

Check if the counterfactual is in the target class.
"""
function in_target_class(ce::CounterfactualExplanation)
    return Models.predict_label(ce.M, ce.data, decode_state(ce))[1] == ce.target
end

"""
    converged(ce::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has converged. The search is considered to have converged only if the counterfactual is valid.
"""
function converged(ce::CounterfactualExplanation)
    return converged(ce, ce.convergence)
end

"""
    converged(ce::CounterfactualExplanation, convergence::DecisionThresholdConvergence)

Checks if the counterfactual search has converged when the convergence criterion is the decision threshold.
"""
function converged(ce::CounterfactualExplanation, convergence::DecisionThresholdConvergence)
    return threshold_reached(ce)
end

"""
    converged(ce::CounterfactualExplanation, convergence::GeneratorConditionsConvergence)

Checks if the counterfactual search has converged when the convergence criterion is generator_conditions.
"""
function converged(
    ce::CounterfactualExplanation, convergence::GeneratorConditionsConvergence
)
    return threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
end

"""
    converged(ce::CounterfactualExplanation, convergence::MaxIterConvergence)

Checks if the counterfactual search has converged when the convergence criterion is maximum iterations.
"""
function converged(ce::CounterfactualExplanation, convergence::MaxIterConvergence)
    return false
end

"""
    converged(ce::CounterfactualExplanation, convergence::InvalidationRateConvergence)

Checks if the counterfactual search has converged when the convergence criterion is invalidation rate.
"""
function converged(ce::CounterfactualExplanation, convergence::InvalidationRateConvergence)
    ir = Generators.invalidation_rate(ce)
    label = predict_label(ce.M, ce.data, ce.x′)[1]
    return label == ce.target && convergence.invalidation_rate > ir
end

"""
    converged(ce::CounterfactualExplanation, convergence::EarlyStoppingConvergence)

Checks if the counterfactual search has converged when the convergence criterion is early stopping.
"""
function converged(ce::CounterfactualExplanation, convergence::EarlyStoppingConvergence)
    return steps_exhausted(ce)
end

"""
    threshold_reached(ce::CounterfactualExplanation)

A convenience method that determines if the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(ce::CounterfactualExplanation)
    γ = ce.convergence.decision_threshold
    success_rate = sum(target_probs(ce) .>= γ) / ce.num_counterfactuals
    return success_rate > ce.convergence.min_success_rate
end

"""
    steps_exhausted(ce::CounterfactualExplanation) 

A convenience method that checks if the number of maximum iterations has been exhausted.
"""
function steps_exhausted(ce::CounterfactualExplanation)
    return ce.search[:iteration_count] == ce.convergence.max_iter
end

"""
    total_steps(ce::CounterfactualExplanation)

A convenience method that returns the total number of steps of the counterfactual search.
"""
function total_steps(ce::CounterfactualExplanation)
    return ce.search[:iteration_count]
end
