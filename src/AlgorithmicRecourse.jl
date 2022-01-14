module AlgorithmicRecourse

# Dependencies:
using Flux
using LinearAlgebra

include("models/Models.jl")
using .Models

include("losses/Losses.jl")
using .Losses

include("generators/Generators.jl")
using .Generators
export Generator, GenericGenerator, GreedyGenerator, update_recourse, convergence

include("core.jl")
export generate_recourse

end