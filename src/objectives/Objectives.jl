module Objectives

using ..CounterfactualExplanations

include("loss_functions.jl")
export logitbinarycrossentropy, logitcrossentropy, mse

include("penalties.jl")

end