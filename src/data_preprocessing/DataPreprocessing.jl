module DataPreprocessing

using CategoricalArrays
using CounterfactualExplanations
using ..GenerativeModels
using DataFrames: DataFrames
using Flux: Flux
using MultivariateStats: MultivariateStats
using StatsBase
using Tables
using Random: Random

include("counterfactual_data.jl")
include("utils.jl")
include("data_contraints.jl")

export CounterfactualData
export select_factual, apply_domain_constraints, OutputEncoder, transformable_features

end
