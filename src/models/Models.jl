module Models

using ..CounterfactualExplanations
using ..DataPreprocessing
using Parameters
using Flux
using MLJBase
using NearestNeighborModels
using Plots
using LazyArtifacts
using Serialization
using LaplaceRedux
using MLUtils
using ProgressMeter
using Statistics

include("utils.jl")

include("differentiable/differentiable.jl")

include("plotting/default.jl")
include("plotting/voronoi.jl")

include("pretrained/pretrained.jl")

export AbstractFittedModel, AbstractDifferentiableModel
export Linear, FluxModel, FluxEnsemble, LaplaceReduxModel
export flux_training_params
export probs, logits

"""
    logits(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the raw model predictions. In classification this is sometimes referred to as *logits*: the non-normalized predictions that are fed into a link function to produce predicted probabilities. In regression (not currently implemented) raw outputs typically correspond to final outputs. In other words, there is typically no normalization involved.
"""
function logits(M::AbstractFittedModel, X::AbstractArray) end

"""
    probs(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the normalized model predictions, so the predicted probabilities in the case of classification. In regression (not currently implemented) this method is redundant. 
"""
function probs(M::AbstractFittedModel, X::AbstractArray) end

"""
    model_catalogue

A dictionary containing all trainable machine learning models.
"""
const model_catalogue = Dict(
    :Linear => Linear, :MLP => FluxModel, :DeepEnsemble => FluxEnsemble
)

"""
    fit_model(
        counterfactual_data::CounterfactualData, model::Symbol=:MLP;
        kwrgs...
    )

Fits one of the available default models to the `counterfactual_data`. The `model` argument can be used to specify the desired model. The available values correspond to the keys of the [`model_catalogue`](@ref) dictionary.
"""
function fit_model(counterfactual_data::CounterfactualData, model::Symbol=:MLP; kwrgs...)
    @assert model in keys(model_catalogue) "Specified model does not match any of the models available in the `model_catalogue`."

    # Set up:
    M = model_catalogue[model](counterfactual_data; kwrgs...)

    # Train:
    train(M, counterfactual_data)

    return M
end

export model_catalogue, fit_model, model_evaluation, predict_label, predict_proba, reset!

end
