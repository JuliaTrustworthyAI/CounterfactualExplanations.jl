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
export ClaPROARGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export REVISEGenerator
export DiCEGenerator
export generator_catalogue
export generate_perturbations, conditions_satisified, mutability_constraints
export Generator, @objective, @threshold

include("functions.jl")
include("macros.jl")

# Gradient-Based Generators:
include("gradient_based/base.jl")
include("gradient_based/functions.jl")
include("gradient_based/generators.jl")
include("gradient_based/optimisers.jl")

"A dictionary countaining the contructors of all available counterfactual generators."
generator_catalogue = Dict(
    :claproar => Generators.ClaPROARGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
)

end
