module Benchmark

using ..Counterfactuals
using DataFrames

export benchmark

include("functions.jl")

function benchmark(counterfactual_explanation::Union{CounterfactualExplanation,Vector{CounterfactualExplanation}};to_dataframe=true)
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