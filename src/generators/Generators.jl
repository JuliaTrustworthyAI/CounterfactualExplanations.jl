# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..Models
using ..Losses
using ..GenerativeModels
using ..CounterfactualState
using Flux
using LinearAlgebra

include("base.jl")

export AbstractGenerator, AbstractGradientBasedGenerator, GenericGenerator, GreedyGenerator,
    generate_perturbations, conditions_satisified, mutability_constraints   

end