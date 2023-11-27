using .DataPreprocessing
using .GenerativeModels
using .Generators
using .Models
using ChainRulesCore
using Flux
using MLUtils
using MultivariateStats
using Statistics
using StatsBase

include("core_struct.jl")
include("convergence_struct.jl")
include("convergence.jl")
include("encodings.jl")
include("generate_counterfactual.jl")
include("growing_spheres.jl")
include("info_extraction.jl")
include("initialisation.jl")
include("latent_space_mappings.jl")
include("path_tracking.jl")
include("printing.jl")
include("search.jl")
include("utils.jl")
include("vectorised.jl")

"""
    convergence_catalogue

A dictionary containing all convergence criteria.
"""
const convergence_catalogue = Dict(
    :decision_threshold => DecisionThresholdConvergence(),
    :generator_conditions => GeneratorConditionsConvergence(),
    :max_iter => MaxIterConvergence(),
    :invalidation_rate => InvalidationRateConvergence(),
    :early_stopping => EarlyStoppingConvergence(),
)
