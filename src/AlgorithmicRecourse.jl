module AlgorithmicRecourse

# Dependencies:
using Flux
using LinearAlgebra

include("models/Models.jl")
using .Models

include("generators/Generators.jl")
using .Generators
export Generator, GenericGenerator

include("losses/Losses.jl")
using .Losses

include("core.jl")
export generate_recourse

end