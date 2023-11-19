"""
    terminated(ce::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has terminated.
"""
function terminated(ce::AbstractCounterfactualExplanation)
    if ce.M isa Models.TreeModel
        return in_target_class(ce)
    end
    return converged(ce) || steps_exhausted(ce)
end

"""
    in_target_class(ce::CounterfactualExplanation)

Check if the counterfactual is in the target class.
"""
function in_target_class(ce::AbstractCounterfactualExplanation)
    return Models.predict_label(ce.M, ce.data, decode_state(ce))[1] == ce.target
end

"""
    converged(ce::CounterfactualExplanation)

A convenience method to determine if the counterfactual search has converged. The search is considered to have converged only if the counterfactual is valid.
"""
function converged(ce::AbstractCounterfactualExplanation)
    if ce.generator isa GrowingSpheresGenerator
        conv = ce.search[:converged]
    elseif ce.convergence[:converge_when] == :decision_threshold
        conv = threshold_reached(ce)
    elseif ce.convergence[:converge_when] == :generator_conditions
        conv = threshold_reached(ce) && Generators.conditions_satisfied(ce.generator, ce)
    elseif ce.convergence[:converge_when] == :max_iter
        conv = false
    elseif ce.convergence[:converge_when] == :invalidation_rate
        ir = Generators.invalidation_rate(ce)
        # gets the label from an array, not sure why it is an array though.
        label = predict_label(ce.M, ce.data, ce.x′)[1]
        conv = label == ce.target && ce.params[:invalidation_rate] > ir
    elseif (ce.convergence[:converge_when] == :early_stopping)
        conv = steps_exhausted(ce)
    else
        @error "Convergence criterion not recognized."
    end

    return conv
end

"""
    threshold_reached(ce::CounterfactualExplanation)

A convenience method that determines if the predefined threshold for the target class probability has been reached.
"""
function threshold_reached(ce::CounterfactualExplanation)
    γ = ce.convergence[:decision_threshold]
    success_rate = sum(target_probs(ce) .>= γ) / ce.num_counterfactuals
    return success_rate > ce.convergence[:min_success_rate]
end

"""
    threshold_reached(ce::CounterfactualExplanation, x::AbstractArray)

A convenience method that determines if the predefined threshold for the target class probability has been reached for a specific sample `x`.
"""
function threshold_reached(ce::CounterfactualExplanation, x::AbstractArray)
    γ = ce.convergence[:decision_threshold]
    success_rate = sum(target_probs(ce, x) .>= γ) / ce.num_counterfactuals
    return success_rate > ce.convergence[:min_success_rate]
end

"""
    steps_exhausted(ce::CounterfactualExplanation) 

A convenience method that checks if the number of maximum iterations has been exhausted.
"""
function steps_exhausted(ce::CounterfactualExplanation)
    return ce.search[:iteration_count] == ce.convergence[:max_iter]
end

"""
    total_steps(ce::CounterfactualExplanation)

A convenience method that returns the total number of steps of the counterfactual search.
"""
function total_steps(ce::CounterfactualExplanation)
    return ce.search[:iteration_count]
end
