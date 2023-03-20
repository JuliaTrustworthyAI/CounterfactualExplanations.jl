module Objectives

using ..CounterfactualExplanations

# Loss functions:
include("loss_functions.jl")
export logitbinarycrossentropy, logitcrossentropy, mse

# Catalogue:
const losses_catalogue = Dict(
    :logitbinarycrossentropy => logitbinarycrossentropy,
    :logitcrossentropy => logitcrossentropy,
    :mse => mse,
)
export losses_catalogue

# Penalities
include("penalties.jl")
export distance, distance_l0, distance_l1, distance_l2, distance_linf
export ddp_diversity

# Catalogue:
const penalties_catalogue = Dict(
    :distance_l0 => distance_l0,
    :distance_l1 => distance_l1,
    :distance_l2 => distance_l2,
    :distance_linf => distance_linf,
    :ddp_diversity => ddp_diversity,
)
export penalties_catalogue

end
