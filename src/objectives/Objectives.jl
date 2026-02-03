module Objectives

using ..CounterfactualExplanations
using Flux: Flux
using Flux.Losses: Losses, logitbinarycrossentropy, logitcrossentropy
using ChainRulesCore: ChainRulesCore
using LinearAlgebra
using Statistics
using Random

include("distance_utils.jl")
include("penalties.jl")
include("traits.jl")

export logitbinarycrossentropy, logitcrossentropy, predictive_entropy
export losses_catalogue
export distance, distance_mad, distance_l0, distance_l1, distance_l2, distance_linf
export distance_cosine, distance_from_target, distance_from_target_cosine
export ddp_diversity
export EnergyDifferential
export hinge_loss
export penalties_catalogue

const losses_catalogue = Dict(
    :logitbinarycrossentropy => logitbinarycrossentropy,
    :logitcrossentropy => logitcrossentropy,
)

const penalties_catalogue = Dict(
    :distance_mad => distance_mad,
    :distance_l0 => distance_l0,
    :distance_l1 => distance_l1,
    :distance_l2 => distance_l2,
    :distance_linf => distance_linf,
    :distance_cosine => distance_cosine,
    :ddp_diversity => ddp_diversity,
    :distance_from_target => distance_from_target,
    :distance_from_target_cosine => distance_from_target_cosine,
    :energy_constraint => energy_constraint,
    :energy_differential => EnergyDifferential(),
    :hinge_loss => hinge_loss,
)

end

"""
    predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)

Computes the predictive entropy of the counterfactuals.
Explained in https://arxiv.org/abs/1406.2541.
"""
function predictive_entropy(ce::AbstractCounterfactualExplanation; agg=Statistics.mean)
    model = ce.M
    counterfactual_data = ce.data
    X = CounterfactualExplanations.decode_state(ce)
    p = CounterfactualExplanations.Models.predict_proba(model, counterfactual_data, X)
    output = -agg(sum(@.(p * log(p)); dims=2))
    return output
end
