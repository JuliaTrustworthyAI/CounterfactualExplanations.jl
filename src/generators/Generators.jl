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
export AbstractGradientBasedGenerator
export AbstractNonGradientBasedGenerator
export ClaPROARGenerator
export FeatureTweakGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export REVISEGenerator
export DiCEGenerator
export WachterGenerator
export feature_tweaking
export generator_catalogue
export generate_perturbations, conditions_satisfied, mutability_constraints
export GradientBasedGenerator
export HeuristicBasedGenerator
export @objective, @threshold, @with_optimiser, @search_feature_space, @search_latent_space
export JSMADescent

include("functions.jl")
include("macros.jl")

# Gradient-Based Generators:
include("gradient_based/base.jl")
include("gradient_based/functions.jl")
include("gradient_based/generators.jl")
include("gradient_based/optimisers.jl")
include("gradient_based/probe.jl")
export ProbeGenerator
export hinge_loss, invalidation_rate

# Non-Gradient-Based Generators:
include("non_gradient_based/base.jl")
include("non_gradient_based/functions.jl")
include("non_gradient_based/generators.jl")

"A dictionary containing the constructors of all available counterfactual generators."
generator_catalogue = Dict(
    :claproar => Generators.ClaPROARGenerator,
    :feature_tweak => Generators.FeatureTweakGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
    :wachter => Generators.WachterGenerator,
    :probe => ProbeGenerator,
)

end
