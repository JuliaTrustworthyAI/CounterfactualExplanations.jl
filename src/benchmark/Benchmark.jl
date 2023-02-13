module Benchmark
using DataFrames
using ..CounterfactualExplanations

export benchmark

include("functions.jl")

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
    to_dataframe = true,
)
    bmk = Dict(
        :success_rate => success_rate(counterfactual_explanation),
        :distance => distance(counterfactual_explanation),
        :redundancy => redundancy(counterfactual_explanation),
    )
    if to_dataframe
        bmk = DataFrame(bmk)
    end
    return bmk
end

end
