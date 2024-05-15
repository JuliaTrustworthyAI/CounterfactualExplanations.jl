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

include("utils.jl")

include("core_struct.jl")
include("traits.jl")

include("mlj.jl")

include("differentiable/differentiable.jl")
include("nondifferentiable/nondifferentiable.jl")

include("pretrained/pretrained.jl")

export AbstractModel
export AbstractDifferentiableModel
export Linear
export MLP
export DeepEnsemble
export FluxModel
export FluxEnsemble
export DecisionTreeModel
export RandomForestModel
export flux_training_params
export probs, logits

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
    :MLP => MLP,
    :DeepEnsemble => DeepEnsemble,
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

export standard_models_catalogue
export all_models_catalogue
export mlj_models_catalogue
export fit_model
export model_evaluation
export predict_label
export predict_proba

end
