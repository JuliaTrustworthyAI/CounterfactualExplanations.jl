"""
	Flux.Losses.logitbinarycrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitbinarycrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitbinarycrossentropy(
    ce::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.logitbinarycrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation)

Simply extends the `logitcrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitcrossentropy(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.logitcrossentropy(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

"""
	Flux.Losses.mse(ce::AbstractCounterfactualExplanation)

Simply extends the `mse` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.mse(ce::AbstractCounterfactualExplanation; kwargs...)
    loss = Flux.Losses.mse(
        logits(ce.M, CounterfactualExplanations.decode_state(ce)),
        ce.target_encoded;
        kwargs...,
    )
    return loss
end

function predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    model = ce.M
    counterfactual_data = ce.data
    X = CounterfactualExplanations.decode_state(ce)
    p = CounterfactualExplanations.Models.predict_proba(model, counterfactual_data, X)
    output = -agg(sum(@.(p * log(p)); dims=2))
    return output
end
