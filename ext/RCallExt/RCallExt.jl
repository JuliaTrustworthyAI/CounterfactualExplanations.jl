module RCallExt

using CounterfactualExplanations
using Flux
using RCall

include("utils.jl")
include("models.jl")
include("generators.jl")

export RTorchModel, rtorch_model_loader

end
