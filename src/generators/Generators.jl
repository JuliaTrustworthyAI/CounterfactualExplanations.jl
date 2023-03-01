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

export AbstractGradientBasedGenerator
export ClaPROARGenerator, ClaPROARGeneratorParams
export GenericGenerator, GenericGeneratorParams
export GravitationalGenerator, GravitationalGeneratorParams
export GreedyGenerator, GreedyGeneratorParams
export REVISEGenerator, REVISEGeneratorParams
export DiCEGenerator, DiCEGeneratorParams
export generator_catalog
export generate_perturbations, conditions_satisified, mutability_constraints

# Loss:
"""
    ℓ(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator loss function to the current counterfactual state for any generator.
"""
function ℓ(
    generator::AbstractGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)

    loss_fun =
        !isnothing(generator.loss) ? generator.loss :
        CounterfactualExplanations.guess_loss(counterfactual_explanation)
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(counterfactual_explanation)
    return loss
end

# Complexity:
"""
    h(generator::AbstractGenerator, counterfactual_explanation::AbstractCounterfactualExplanation)

The default method to apply the generator complexity penalty to the current counterfactual state for any generator.
"""
function h(
    generator::AbstractGenerator,
    counterfactual_explanation::AbstractCounterfactualExplanation,
)
    dist_ = generator.complexity(
        counterfactual_explanation.x .-
        CounterfactualExplanations.decode_state(counterfactual_explanation),
    )
    penalty = generator.λ * dist_
    return penalty
end

include("gradient_based/functions.jl")

generator_catalog = Dict(
    :claproar => Generators.ClaPROARGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
)

end
