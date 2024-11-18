module Evaluation

using ..CounterfactualExplanations
using DataFrames
using ..Generators
using ..Models
using LinearAlgebra: LinearAlgebra
using Statistics

include("benchmark.jl")
include("evaluate.jl")
include("measures.jl")

export Benchmark, benchmark, evaluate, default_measures
export validity, redundancy
export plausibility
export plausibility_energy_differential, plausibility_cosine, plausibility_distance_from_target
export faithfulness
export plausibility_measures, default_measures, distance_measures, all_measures
export concatenate_benchmarks

"Available plausibility measures."
const plausibility_measures = [
    plausibility_energy_differential, plausibility_cosine, plausibility_distance_from_target
]

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

"All measures."
const all_measures = [
    validity, 
    redundancy,
    collect(values(CounterfactualExplanations.Objectives.penalties_catalogue))...,
    plausibility_measures...,
    faithfulness,
]

end
