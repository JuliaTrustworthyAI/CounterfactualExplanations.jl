module Models

using ..CounterfactualExplanations
using ..DataPreprocessing
using DataFrames: DataFrames
using Flux
using LazyArtifacts: LazyArtifacts
using MLJBase: MLJBase
using MLUtils
using ProgressMeter
using Serialization
using Statistics: Statistics
using MLJDecisionTreeInterface: MLJDecisionTreeInterface

include("utils.jl")

include("differentiable/differentiable.jl")
include("nondifferentiable/nondifferentiable.jl")

include("pretrained/pretrained.jl")

export AbstractModel
export AbstractDifferentiableModel
export Linear
export FluxModel
export FluxEnsemble
export DecisionTreeModel
export RandomForestModel
export flux_training_params
export probs, logits

abstract type AbstractModelType end

"""
    Model <: AbstractModel

Constructor for all models.
"""
mutable struct Model <: AbstractModel
    model
    likelihood::Symbol
    fitresult
    type::AbstractModelType
end

"""
    Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)

Outer constructor for `Model`.
"""
function Model(model, type::AbstractModelType; likelihood::Symbol=:classification_binary)
    return Model(model, likelihood, nothing, type)
end

"""
    logits(M::Model, X::AbstractArray)

Returns the logits of the model.
"""
logits(M::Model, X::AbstractArray) = logits(M, M.type, X)

"""
    probs(M::Model, X::AbstractArray)

Returns the probabilities of the model.
"""
probs(M::Model, X::AbstractArray) = probs(M, M.type, X)

"""
    (M::Model)(data::CounterfactualData; kwargs...)

Wrap model `M` around the data in `data`.
"""
function (M::Model)(data::CounterfactualData; kwargs...)
    return (M::Model)(data, M.type; kwargs...)
end

"""
    standard_models_catalogue

A dictionary containing all differentiable machine learning models.
"""
const standard_models_catalogue = Dict(
    :Linear => Linear, :MLP => FluxModel, :DeepEnsemble => FluxEnsemble
)

"""
    all_models_catalogue

A dictionary containing both differentiable and non-differentiable machine learning models.
"""
const all_models_catalogue = Dict(
    :Linear => Linear,
    :MLP => FluxModel,
    :DeepEnsemble => FluxEnsemble,
    :DecisionTree => DecisionTreeModel,
    :RandomForest => RandomForestModel,
)

"""
    mlj_models_catalogue

A dictionary containing all machine learning models from the MLJ model registry that the package supports.
"""
const mlj_models_catalogue = Dict(
    :DecisionTree => DecisionTreeModel, :RandomForest => RandomForestModel
)

function fit_model(
    counterfactual_data::CounterfactualData, type::AbstractModelType; kwrgs...
)
end

"""
    fit_model(
        counterfactual_data::CounterfactualData, model::Symbol=:MLP;
        kwrgs...
    )

Fits one of the available default models to the `counterfactual_data`. The `model` argument can be used to specify the desired model. The available values correspond to the keys of the [`all_models_catalogue`](@ref) dictionary.
"""
function fit_model(counterfactual_data::CounterfactualData, model::Symbol=:MLP; kwrgs...)
    @assert model in keys(all_models_catalogue) "Specified model does not match any of the models available in the `all_models_catalogue`."

    # Set up:
    M = all_models_catalogue[model](counterfactual_data; kwrgs...)
    M = train(M, counterfactual_data)

    return M
end

"""
    fit_model(
        counterfactual_data::CounterfactualData,
        model::Union{Type{<:AbstractModel},Function};
        kwrgs...,
    )

Fits a custom model to the `counterfactual_data`. The `model` argument should be an instance of a custom model.
"""
function fit_model(
    counterfactual_data::CounterfactualData,
    model::Union{Type{<:AbstractModel},Function};
    kwrgs...,
)
    M = model(counterfactual_data; kwrgs...)
    M = train(M, counterfactual_data)
    return M
end

export standard_models_catalogue
export all_models_catalogue
export mlj_models_catalogue
export fit_model
export model_evaluation
export predict_label
export predict_proba

end
