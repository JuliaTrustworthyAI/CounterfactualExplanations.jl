"""
    path(ce::CounterfactualExplanation)

A convenience method that returns the entire counterfactual path.
"""
function path(ce::CounterfactualExplanation; feature_space=true)
    path = deepcopy(ce.search[:path])
    if feature_space
        path = [decode_state(ce, z) for z in path]
    end
    return path
end

"""
    counterfactual_probability_path(ce::CounterfactualExplanation)

Returns the counterfactual probabilities for each step of the search.
"""
function counterfactual_probability_path(ce::CounterfactualExplanation)
    M = ce.M
    p = map(X -> counterfactual_probability(ce, X), path(ce))
    return p
end

"""
    counterfactual_label_path(ce::CounterfactualExplanation)

Returns the counterfactual labels for each step of the search.
"""
function counterfactual_label_path(ce::CounterfactualExplanation)
    counterfactual_data = ce.data
    M = ce.M
    ŷ = map(X -> predict_label(M, counterfactual_data, X), path(ce))
    return ŷ
end

"""
    target_probs_path(ce::CounterfactualExplanation)

Returns the target probabilities for each step of the search.
"""
function target_probs_path(ce::CounterfactualExplanation)
    X = path(ce)
    P = map(x -> target_probs(ce, x), X)
    return P
end

"""
    embed_path(ce::CounterfactualExplanation)

Helper function that embeds path into two dimensions for plotting.
"""
function embed_path(ce::CounterfactualExplanation)
    data_ = ce.data
    return DataPreprocessing.embed(data_, path(ce))
end
