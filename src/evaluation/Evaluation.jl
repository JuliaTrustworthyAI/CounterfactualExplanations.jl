module Evaluation

using ..CounterfactualExplanations
using DataFrames
using ..Models
using LinearAlgebra
using Statistics

include("benchmark.jl")
include("evaluate.jl")
include("measures.jl")

export Benchmark, benchmark, evaluate, default_measures
export validity, distance, redundancy

"The default evaluation measures."
const default_measures = [
    validity, CounterfactualExplanations.Objectives.distance, redundancy
]

"All distance measures."
const distance_measures = [
    CounterfactualExplanations.Objectives.distance_l0,
    CounterfactualExplanations.Objectives.distance_l1,
    CounterfactualExplanations.Objectives.distance_l2,
    CounterfactualExplanations.Objectives.distance_linf,
]

end
