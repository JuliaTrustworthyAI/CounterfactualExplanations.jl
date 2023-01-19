module Models

using ..CounterfactualExplanations
using ..DataPreprocessing
using Parameters

export AbstractFittedModel, AbstractDifferentiableModel
export FluxModel, FluxEnsemble, LaplaceReduxModel
export probs, logits

"""
Base type for fitted models.
"""
abstract type AbstractFittedModel end

"""
    logits(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the raw model predictions. In classification this is sometimes referred to as *logits*: the non-normalized predictions that are fed into a link function to produce predicted probabilities. In regression (not currently implemented) raw outputs typically correspond to final outputs. In other words, there is typically no normalization involved.
"""
function logits(M::AbstractFittedModel, X::AbstractArray) end

"""
    probs(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the normalized model predictions, so the predicted probabilities in the case of classifiation. In regression (not currently implemented) this method is redundant. 
"""
function probs(M::AbstractFittedModel, X::AbstractArray) end

include("differentiable/differentiable.jl")
include("plotting.jl")

"""
    model_catalogue

A dictionary containing all trainable machine learning models.
"""
const model_catalogue = Dict(
    :LogisticRegression => LogisticRegression,
    :MLP => FluxModel,
    :DeepEnsemble => FluxEnsemble,
)

function fit_model(
    counterfactual_data::CounterfactualData, model::Symbol=:MLP
)
    @assert model in keys(model_catalogue) "Specified model does not match any of the models available in the `model_catalogue`."

    # Set up:
    M = model_catalogue[model](counterfactual_data)

    # Train:
    train(M, counterfactual_data)

    return M
end

export model_catalogue, fit_model

end
