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

include("utils.jl")
export plot_data!, plot_contour, toy_data_linear, toy_data_non_linear, build_model, build_ensemble

end