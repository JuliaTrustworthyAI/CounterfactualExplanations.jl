module Losses

using Flux.Losses: logitcrossentropy, logitbinarycrossentropy, mse, mae

export logitcrossentropy, logitbinarycrossentropy, mse, mae, hinge_loss

include("functions.jl")

end