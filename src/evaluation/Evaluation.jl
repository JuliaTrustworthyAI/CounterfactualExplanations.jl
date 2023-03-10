module Evaluation

using ..CounterfactualExplanations
using DataFrames
using Statistics

export Benchmark, benchmark, evaluate, default_measures
export validity, distance, redundancy

include("measures.jl")

"The default evaluation measures."
const default_measures = [
    validity,
    distance,
    redundancy
]

"All distance measures."
const distance_measures = [
    distance_l0,
    distance_l1,
    distance_l2,
    distance_linf
]

"""
    evaluate(
        counterfactual_explanation::CounterfactualExplanation;
        measure::Union{Function,Vector{Function}}=default_measures,
        agg::Function=mean,
        report_each::Bool=false,
        output_format::Symbol=:Vector,
        pivot_longer::Bool=true
    )

Just computes evaluation `measures` for the counterfactual explanation.
"""
function evaluate(
    counterfactual_explanation::CounterfactualExplanation;
    measure::Union{Function,Vector{Function}}=default_measures,
    agg::Function=mean,
    report_each::Bool=false,
    output_format::Symbol=:Vector,
    pivot_longer::Bool=true
)
    # Setup:
    @assert output_format ∈ [:Vector, :Dict, :DataFrame]
    measure = typeof(measure) <: Function ? [measure] : measure
    agg = report_each ? (x -> x) : agg
    function _compute_measure(ce, fun)
        val = agg(fun(ce))
        val = ndims(val) > 1 ? vec(val) : [val]
        return val
    end

    # Evaluate:
    evaluation = [_compute_measure(counterfactual_explanation, fun) for fun in measure]

    # As Dict:
    if output_format == :Dict
        evaluation = Dict(m => ndims(val) > 1 ? vec(val) : val for (m, val) in zip(Symbol.(measure), evaluation))
    end

    # As DataFrame:
    if output_format == :DataFrame
        evaluation = Dict(m => ndims(val) > 1 ? vec(val) : val for (m, val) in zip(Symbol.(measure), evaluation)) |> DataFrame
        evaluation.num_counterfactual = 1:nrow(evaluation)
        if pivot_longer
            evaluation = stack(evaluation, Not(:num_counterfactual))
        end
        select!(evaluation, :num_counterfactual, :)
    end

    return evaluation
end

"""
    evaluate(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        report_meta::Bool=false,
        meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
        kwargs...
    )

Computes evaluation `measures` for a vector of counterfactual explanations. By default, no meta data is reported. For `report_meta=true`, meta data is automatically inferred, unless this overwritted by `meta_data`. The optional `meta_data` argument should be a vector of dictionaries of the same length as the vector of counterfactual explanations. 

Additional `kwargs...` can be provided (see [`evaluate(counterfactual_explanation::CounterfactualExplanation`](@ref) for details).
"""
function evaluate(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    report_meta::Bool=false,
    meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
    kwargs...
)
    evaluations = []
    for (i, ce) in enumerate(counterfactual_explanations)
        evaluation = evaluate(ce; output_format=:DataFrame, kwargs...)
        if report_meta || !isnothing(meta_data)
            if !isnothing(meta_data)
                @assert length(counterfactual_explanations) == length(meta_data)
                df_meta = DataFrame(meta_data[i])
            else
                df_meta = DataFrame(CounterfactualExplanations.get_meta(ce))
            end
            if !("sample" ∈ names(df_meta)) 
                df_meta.sample .= i
            end
            evaluation = crossjoin(evaluation, df_meta, makeunique=true)
            evaluation.target .= ce.target
            evaluation.factual .= CounterfactualExplanations.factual_label(ce)
        end
        if !("sample" ∈ names(evaluation)) 
            evaluation.sample .= i
        end
        evaluations = [evaluations..., evaluation]
    end
    evaluations = reduce(vcat, evaluations)
    select!(evaluations, :sample, :num_counterfactual, :)
    return evaluations

end

include("benchmark.jl")

end