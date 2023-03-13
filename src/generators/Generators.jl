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
export ClaPROARGenerator, ClaPROARGeneratorParams
export GenericGenerator, GenericGeneratorParams
export GravitationalGenerator, GravitationalGeneratorParams
export GreedyGenerator, GreedyGeneratorParams
export REVISEGenerator, REVISEGeneratorParams
export DiCEGenerator, DiCEGeneratorParams
export generator_catalogue
export generate_perturbations, conditions_satisified, mutability_constraints
export ComposableGenerator, @objective, @threshold

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
    counterfactual_explanation::AbstractCounterfactualExplanation
)
    if isnothing(generator.complexity)
        penalty = 0.0
    elseif typeof(generator.complexity) <: Vector
        cost = [fun(counterfactual_explanation) for fun in generator.complexity]
    else
        cost = generator.complexity(counterfactual_explanation)
    end
    penalty = sum(generator.λ .* cost)
    return penalty
end

include("gradient_based/functions.jl")

"A dictionary countaining the contructors of all available counterfactual generators."
generator_catalogue = Dict(
    :claproar => Generators.ClaPROARGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
)

include("ComposableGenerator.jl")

end
