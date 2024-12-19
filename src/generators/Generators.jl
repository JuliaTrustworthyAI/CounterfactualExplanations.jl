# generators.jl
#
# Core package functionality that implements algorithmic recourse.
module Generators

using ..CounterfactualExplanations
using ..GenerativeModels
using Flux: Flux
using LinearAlgebra
using ..Models
using ..Convergence
using ..Objectives
using Statistics: Statistics
using DataFrames: DataFrames
using MLJBase: MLJBase
using Distributions: Distributions
using Random

export AbstractGradientBasedGenerator
export AbstractNonGradientBasedGenerator
export ClaPROARGenerator
export CLUEGenerator
export DiCEGenerator
export ECCoGenerator
export FeatureTweakGenerator
export GenericGenerator
export GravitationalGenerator
export GreedyGenerator
export GrowingSpheresGenerator
export REVISEGenerator
export WachterGenerator
export FeatureTweakGenerator
export generator_catalogue
export generate_perturbations
export GradientBasedGenerator
export @objective, @with_optimiser, @search_feature_space, @search_latent_space
export JSMADescent
export predictive_entropy
export ProbeGenerator

include("macros.jl")
include("loss.jl")
include("complexity.jl")
include("generate_perturbations.jl")

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
    :ecco => Generators.ECCoGenerator,
)

"""
    incompatible(AbstractGenerator, AbstractCounterfactualExplanation)

Checks if the generator is incompatible with any of the additional specifications for the counterfactual explanations. By default, generators are assumed to be compatible.
"""
function incompatible(AbstractGenerator, AbstractCounterfactualExplanation)
    return false
end

"""
    total_loss(ce::AbstractCounterfactualExplanation)

Computes the total loss of a counterfactual explanation with respect to the search objective.
"""
total_loss(ce::AbstractCounterfactualExplanation) =
    if hasfield(typeof(ce.generator), :loss)
        ℓ(ce.generator, ce) + h(ce.generator, ce)
    else
        nothing
    end

end
