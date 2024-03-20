# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..CounterfactualExplanations
using ..GenerativeModels
using Flux
using LinearAlgebra
using ..Models
using ..Objectives
using Statistics
using Parameters
using DecisionTree
using DataFrames
using MLJBase
using MLJDecisionTreeInterface
using Distributions
using Random
using Statistics

export AbstractGradientBasedGenerator
export AbstractNonGradientBasedGenerator
export ClaPROARGenerator
export CLUEGenerator
export DiCEGenerator
export FeatureTweakGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export GrowingSpheresGenerator
export growing_spheres_shrink!, growing_spheres_expand!, feature_selection!
export REVISEGenerator
export WachterGenerator
export FeatureTweakGenerator
export feature_tweaking!
export generator_catalogue
export generate_perturbations, conditions_satisfied
export GradientBasedGenerator
export @objective, @with_optimiser, @search_feature_space, @search_latent_space
export JSMADescent
export hinge_loss
export predictive_entropy
export ProbeGenerator

include("macros.jl")
include("loss.jl")
include("complexity.jl")

# Optimizers
include("optimizers/JSMADescent.jl")

# Gradient-Based Generators:
include("gradient_based/base.jl")
include("gradient_based/generate_perturbations.jl")
include("gradient_based/generators.jl")
include("gradient_based/loss.jl")
include("gradient_based/utils.jl")

# Non-Gradient-Based Generators:
include("non_gradient_based/base.jl")

include("non_gradient_based/feature_tweak/generate_perturbations.jl")
include("non_gradient_based/growing_spheres/growing_spheres.jl")
include("non_gradient_based/utils.jl")

"A dictionary containing the constructors of all available counterfactual generators."
generator_catalogue = Dict(
    :claproar => Generators.ClaPROARGenerator,
    :feature_tweak => Generators.FeatureTweakGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :growing_spheres => Generators.GrowingSpheresGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
    :wachter => Generators.WachterGenerator,
    :probe => Generators.ProbeGenerator,
    :clue => Generators.CLUEGenerator,
)

end
