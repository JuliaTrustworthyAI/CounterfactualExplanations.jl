using DataFrames

struct Benchmark
    counterfactual_explanations::Vector{CounterfactualExplanation}
    evaluation::DataFrame
end

"""
    benchmark(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
        measure::Union{Function,Vector{Function}}=default_measures
    )

Generates a `Benchmark` for a vector of counterfactual explanations. Optionally `meta_data` describing each individual counterfactual explanation can be supplied. This should be a vector of dictionaries of the same length as the vector of counterfactuals. If no `meta_data` is supplied, it will be automatically inferred. 
"""
function benchmark(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures
)
    evaluations = evaluate(counterfactual_explanations; measure=measure, report_meta=true, meta_data=meta_data)
    bmk = Benchmark(counterfactual_explanations, evaluations)
    return bmk
end