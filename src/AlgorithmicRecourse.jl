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
export Generator, GenericGenerator, GreedyGenerator, generate_perturbations, conditions_satisified, mutability_constraints

include("core.jl")
export generate_recourse

end