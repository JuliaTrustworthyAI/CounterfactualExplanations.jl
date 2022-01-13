module Losses

using Flux.Losses: logitbinarycrossentropy, mse, mae

export logitbinarycrossentropy, mse, mae, hinge_loss

include("functions.jl")

end