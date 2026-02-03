"""
    predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)

Computes the predictive entropy of the counterfactuals.
Explained in https://arxiv.org/abs/1406.2541.
"""
function predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    model = ce.M
    counterfactual_data = ce.data
    X = CounterfactualExplanations.decode_state(ce)
    p = CounterfactualExplanations.Models.predict_proba(model, counterfactual_data, X)
    output = -agg(sum(@.(p * log(p)); dims=2))
    return output
end
