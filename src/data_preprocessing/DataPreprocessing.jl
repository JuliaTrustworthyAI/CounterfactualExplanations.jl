module DataPreprocessing

using CategoricalArrays
using CounterfactualExplanations
using DataFrames
using Flux
using MultivariateStats
using StatsBase
using Tables
using UMAP
using MLJBase
using Plots
using Random
using ..GenerativeModels

include("counterfactual_data.jl")
include("utils.jl")
include("generative_model_utils.jl")
include("data_contraints.jl")
include("visualization/plotting.jl")

export CounterfactualData
export select_factual, apply_domain_constraints, OutputEncoder, transformable_features

end
