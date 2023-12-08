"""
    function terminated(ce::CounterfactualExplanation)
        
A convenience method that checks if the counterfactual search has terminated.
"""
function terminated(ce::CounterfactualExplanation)
    return Convergence.converged(ce.convergence, ce) || steps_exhausted(ce)
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

"""
    in_target_class(ce::CounterfactualExplanation)

Check if the counterfactual is in the target class.
"""
function in_target_class(ce::AbstractCounterfactualExplanation)
    return Models.predict_label(ce.M, ce.data, decode_state(ce))[1] == ce.target
end