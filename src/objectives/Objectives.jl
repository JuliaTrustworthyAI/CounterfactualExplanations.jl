module Objectives

using ..CounterfactualExplanations
using Flux: Flux
using Flux.Losses
using ChainRulesCore: ChainRulesCore
using LinearAlgebra
using Statistics
using Random

include("distance_utils.jl")
include("loss_functions.jl")
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
export choose_ad_backend

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
