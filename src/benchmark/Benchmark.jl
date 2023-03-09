module Benchmark

using ..CounterfactualExplanations
using DataFrames
using Statistics

export benchmark, evaluate, default_measures
export validity, distance, redundancy

include("measures.jl")
const default_measures = [
    validity,
    distance,
    redundancy
]

"""
    evaluate(
        counterfactual_explanation::Union{
            CounterfactualExplanation,
            Vector{CounterfactualExplanation},
        };
        measure::Union{Function,Vector{Function}} = default_measures,
    )

Just computes evaluation `measures` for the counterfactual explanation.
"""
function evaluate(
    counterfactual_explanation::Union{
        CounterfactualExplanation,
        Vector{CounterfactualExplanation},
    };
    measure::Union{Function,Vector{Function}}=default_measures,
    agg::Function=mean,
    report_each::Bool=false,
)
    if typeof(measure) <: Function
        measure = [measure]
    end
    agg = report_each ? (x -> x) : agg
    return [fun(counterfactual_explanation; agg=agg) for fun in measure]
end

"""
    benchmark(
        counterfactual_explanation::Union{
            CounterfactualExplanation,
            Vector{CounterfactualExplanation},
        };
        to_dataframe = true,
    )

Computes evaluation metrics for a single or multiple counterfactual explantions and returns a dataframe. This functionality is still basic and will be enhanced in the future. 
"""
function benchmark(
    counterfactual_explanation::Union{
        CounterfactualExplanation,
        Vector{CounterfactualExplanation},
    };
    to_dataframe=true
)
    bmk = Dict(
        :validity => validity(counterfactual_explanation),
        :distance => distance(counterfactual_explanation),
        :redundancy => redundancy(counterfactual_explanation),
    )
    if to_dataframe
        bmk = DataFrame(bmk)
    end
    return bmk
end

end
