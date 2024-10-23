module Objectives

using ..CounterfactualExplanations
using Flux
using Flux.Losses: Losses, logitbinarycrossentropy, logitcrossentropy, mse
using ChainRulesCore: ChainRulesCore
using LinearAlgebra
using Statistics
using Random

include("distance_utils.jl")
include("loss_functions.jl")
include("penalties.jl")
include("traits.jl")

export logitbinarycrossentropy, logitcrossentropy, mse, predictive_entropy
export losses_catalogue
export distance, distance_mad, distance_l0, distance_l1, distance_l2, distance_linf
export ddp_diversity
export EnergyDifferential
export penalties_catalogue

const losses_catalogue = Dict(
    :logitbinarycrossentropy => logitbinarycrossentropy,
    :logitcrossentropy => logitcrossentropy,
    :mse => mse,
)

const penalties_catalogue = Dict(
    :distance_mad => distance_mad,
    :distance_l0 => distance_l0,
    :distance_l1 => distance_l1,
    :distance_l2 => distance_l2,
    :distance_linf => distance_linf,
    :ddp_diversity => ddp_diversity,
    :energy_constraint => energy_constraint,
    :energy_differential => EnergyDifferential(),
)

end
