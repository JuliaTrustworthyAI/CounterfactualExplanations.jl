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
export Generator, GenericGenerator, GreedyGenerator, generate_perturbations, conditions_satisified, mutability_constraints

include("generate_recourse.jl")
export generate_recourse

include("utils.jl")
export plot_data!, plot_contour, plot_contour_multi, toy_data_linear, toy_data_non_linear, toy_data_multi, build_model, build_ensemble

include("Data.jl")
using .Data

end