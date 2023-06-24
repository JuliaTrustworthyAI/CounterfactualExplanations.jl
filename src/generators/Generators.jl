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
using SliceMap
using Random
using Statistics
using RCall
using PythonCall

export AbstractGradientBasedGenerator
export AbstractNonGradientBasedGenerator
export ClaPROARGenerator
export FeatureTweakGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export GrowingSpheresGenerator
export REVISEGenerator
export DiCEGenerator
export WachterGenerator
export FeatureTweakGenerator
export feature_tweaking
export feature_selection
export generator_catalogue
export generate_perturbations, conditions_satisfied, mutability_constraints
export GradientBasedGenerator
export @objective, @threshold, @with_optimiser, @search_feature_space, @search_latent_space
export JSMADescent
export hinge_loss, invalidation_rate
export ProbeGenerator
export growing_spheres_generation

include("macros.jl")
include("utils.jl")

# Optimizers
include("optimizers/JSMADescent.jl")

# Gradient-Based Generators:
include("gradient_based/base.jl")
include("gradient_based/utils.jl")
include("gradient_based/loss.jl")

include("gradient_based/generators.jl")
include("gradient_based/probe.jl")

# Non-Gradient-Based Generators:
include("non_gradient_based/base.jl")

include("non_gradient_based/feature_tweak/feature_tweak.jl")
include("non_gradient_based/growing_spheres/growing_spheres.jl")

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
    :probe => ProbeGenerator,
)

end
