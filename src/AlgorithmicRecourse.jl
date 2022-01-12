module AlgorithmicRecourse

# Dependencies:
using Flux
using LinearAlgebra

include("models/Models.jl")
using .Models

include("generators/Generators.jl")
using .Generators
export Generator, GenericGenerator

include("core.jl")
export generate_recourse

export func
"""
    func(x)

Returns double the number `x` plus `1`.
"""
func(x) = 2x + 1

end