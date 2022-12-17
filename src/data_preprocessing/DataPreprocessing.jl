module DataPreprocessing

using ..GenerativeModels
include("transformations.jl")
include("functions.jl")
export CounterfactualData, select_factual, apply_domain_constraints

include("plotting.jl")



end
