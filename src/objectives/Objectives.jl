module Objectives

using ..CounterfactualExplanations
using ..Models
using ..DataPreprocessing
using Flux
using Flux.Losses
using ChainRulesCore
using LinearAlgebra
using Statistics
using Random

include("distance_utils.jl")
include("loss_functions.jl")
include("penalties.jl")

export logitbinarycrossentropy, logitcrossentropy, mse
export losses_catalogue
export distance, distance_mad, distance_l0, distance_l1, distance_l2, distance_linf
export ddp_diversity
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
)

end
