module RCallExt

using CounterfactualExplanations
using RCall

include("utils.jl")
include("models.jl")
include("generators.jl")

export RTorchModel

end