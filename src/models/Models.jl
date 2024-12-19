module Models

using ..CounterfactualExplanations
using ..DataPreprocessing
using DataFrames: DataFrames
using Flux: Flux
using LazyArtifacts: LazyArtifacts
using MLJBase: MLJBase
using MLUtils
using ProgressMeter
using Serialization
using Statistics: Statistics

include("utils.jl")

include("core_struct.jl")
include("traits.jl")

include("mlj.jl")

include("differentiable/differentiable.jl")

include("pretrained/pretrained.jl")

export AbstractModel
export Model
export Linear
export MLP
export DeepEnsemble
export FluxModel
export FluxEnsemble
export flux_training_params
export probs, logits
export load_mnist_mlp, load_fashion_mnist_ensemble, load_fashion_mnist_vae
export load_cifar_10_mlp, load_cifar_10_ensemble, load_cifar_10_vae

"""
    standard_models_catalogue

A dictionary containing all differentiable machine learning models.
"""
const standard_models_catalogue = Dict(
    :Linear => Linear, :MLP => MLP, :DeepEnsemble => DeepEnsemble
)

"""
    all_models_catalogue

A dictionary containing both differentiable and non-differentiable machine learning models.
"""
const all_models_catalogue = Dict(
    :Linear => Linear, :MLP => MLP, :DeepEnsemble => DeepEnsemble
)

export standard_models_catalogue
export all_models_catalogue
export fit_model
export model_evaluation
export predict_label
export predict_proba

end
