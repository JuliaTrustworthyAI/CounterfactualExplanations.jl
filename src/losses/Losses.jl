module Losses

using Flux.Losses: binarycrossentropy, mse, mae

export binarycrossentropy, mse, mae, hinge_loss

include("functions.jl")

end