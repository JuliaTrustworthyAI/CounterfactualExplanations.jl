module Evaluation

using ..CounterfactualExplanations
using DataFrames
using ProgressMeter
using Statistics
using DataFrames
using ..Models
using ProgressMeter
using LinearAlgebra
using SliceMap

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
