using Flux
using Flux.Losses

"""
    Flux.Losses.logitbinarycrossentropy(counterfactual_explanation::AbstractCounterfactualExplanation)

Simply extends the `logitbinarycrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitbinarycrossentropy(
    counterfactual_explanation::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.logitbinarycrossentropy(
        logits(
            counterfactual_explanation.M,
            CounterfactualExplanations.decode_state(counterfactual_explanation),
        ),
        counterfactual_explanation.target_encoded;
        kwargs...,
    )
    return loss
end

"""
    Flux.Losses.logitcrossentropy(counterfactual_explanation::AbstractCounterfactualExplanation)

Simply extends the `logitcrossentropy` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.logitcrossentropy(
    counterfactual_explanation::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.logitcrossentropy(
        logits(
            counterfactual_explanation.M,
            CounterfactualExplanations.decode_state(counterfactual_explanation),
        ),
        counterfactual_explanation.target_encoded;
        kwargs...,
    )
    return loss
end

"""
    Flux.Losses.mse(counterfactual_explanation::AbstractCounterfactualExplanation)

Simply extends the `mse` method to work with objects of type `AbstractCounterfactualExplanation`.
"""
function Flux.Losses.mse(
    counterfactual_explanation::AbstractCounterfactualExplanation; kwargs...
)
    loss = Flux.Losses.mse(
        logits(
            counterfactual_explanation.M,
            CounterfactualExplanations.decode_state(counterfactual_explanation),
        ),
        counterfactual_explanation.target_encoded;
        kwargs...,
    )
    return loss
end
