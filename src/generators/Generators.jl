# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..CounterfactualExplanations
using ..GenerativeModels
using Flux
using LinearAlgebra
using ..Losses
using ..Models

export AbstractGenerator, AbstractGradientBasedGenerator
export ClapROARGenerator, ClapROARGeneratorParams
export GenericGenerator, GenericGeneratorParams
export GravitationalGenerator, GravitationalGeneratorParams
export GreedyGenerator, GreedyGeneratorParams
export REVISEGenerator, REVISEGeneratorParams
export DiCEGenerator, DiCEGeneratorParams
export generator_catalog
export generate_perturbations, conditions_satisified, mutability_constraints

"""
    AbstractGenerator

An abstract type that serves as the base type for counterfactual generators. 
"""
abstract type AbstractGenerator end

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
        !isnothing(generator.loss) ? getfield(Losses, generator.loss) :
        CounterfactualExplanations.guess_loss(counterfactual_explanation)
    @assert !isnothing(loss_fun) "No loss function provided and loss function could not be guessed based on model."
    loss = loss_fun(
        getfield(Models, :logits)(
            counterfactual_explanation.M,
            CounterfactualExplanations.map_from_latent(counterfactual_explanation),
        ),
        counterfactual_explanation.target_encoded,
    )
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
        CounterfactualExplanations.map_from_latent(counterfactual_explanation),
    )
    penalty = generator.λ * dist_
    return penalty
end

include("gradient_based/functions.jl")

generator_catalog = Dict(
    :claproar => Generators.ClapROARGenerator,
    :generic => Generators.GenericGenerator,
    :gravitational => Generators.GravitationalGenerator,
    :greedy => Generators.GreedyGenerator,
    :revise => Generators.REVISEGenerator,
    :dice => Generators.DiCEGenerator,
)

end
