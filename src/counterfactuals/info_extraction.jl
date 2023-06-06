"""
    factual(ce::CounterfactualExplanation)

A convenience method to retrieve the factual `x`.
"""
function factual(ce::CounterfactualExplanation)
    return ce.x
end

"""
    factual_probability(ce::CounterfactualExplanation)

A convenience method to compute the class probabilities of the factual.
"""
function factual_probability(ce::CounterfactualExplanation)
    return Models.probs(ce.M, ce.x)
end

"""
    factual_label(ce::CounterfactualExplanation)  

A convenience method to get the predicted label associated with the factual.
"""
function factual_label(ce::CounterfactualExplanation)
    M = ce.M
    counterfactual_data = ce.data
    y = predict_label(M, counterfactual_data, factual(ce))
    return y
end

"""
    counterfactual(ce::CounterfactualExplanation)

A convenience method that returns the counterfactual.
"""
function counterfactual(ce::CounterfactualExplanation)
    return decode_state(ce)
end

"""
    counterfactual_probability(ce::CounterfactualExplanation)

A convenience method that computes the class probabilities of the counterfactual.
"""
function counterfactual_probability(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)
    if isnothing(x)
        x = counterfactual(ce)
    end
    p = Models.probs(ce.M, x)
    return p
end

"""
    counterfactual_label(ce::CounterfactualExplanation) 

A convenience method that returns the predicted label of the counterfactual.
"""
function counterfactual_label(ce::CounterfactualExplanation)
    M = ce.M
    counterfactual_data = ce.data
    y = predict_label(M, counterfactual_data, counterfactual(ce))
    return y
end

"""
    target_probs(
        ce::CounterfactualExplanation,
        x::Union{AbstractArray,Nothing}=nothing,
    )

Returns the predicted probability of the target class for `x`. If `x` is `nothing`, the predicted probability corresponding to the counterfactual value is returned.
"""
function target_probs(
    ce::CounterfactualExplanation, x::Union{AbstractArray,Nothing}=nothing
)
    data = ce.data
    likelihood = ce.data.likelihood
    p = counterfactual_probability(ce, x)
    target = ce.target
    target_idx = get_target_index(data.y_levels, target)
    if likelihood == :classification_binary
        if target_idx == 2
            p_target = p
        else
            p_target = 1 .- p
        end
    else
        p_target = ChainRulesCore.selectdim(p, 1, target_idx)
    end
    return p_target
end
