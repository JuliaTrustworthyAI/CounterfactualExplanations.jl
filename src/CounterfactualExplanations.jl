module CounterfactualExplanations

# Dependencies:
using Flux
using LinearAlgebra

include("models/Models.jl")
using .Models

include("losses/Losses.jl")
using .Losses

include("generators/Generators.jl")
using .Generators
export Generator, GenericGenerator, GreedyGenerator, 
    generate_perturbations, conditions_satisified, mutability_constraints

include("data/Data.jl")
using .Data

include("generate_counterfactual.jl")
export generate_counterfactual

end