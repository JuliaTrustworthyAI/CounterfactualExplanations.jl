module AlgorithmicRecourse

# Dependencies:
using Flux
using LinearAlgebra

# Exported functions:
export 
    generate_recourse

# Scripts:
include("models/Models.jl")
using .Models

include("generators/Generators.jl")
using .Generators

include("core.jl")

end