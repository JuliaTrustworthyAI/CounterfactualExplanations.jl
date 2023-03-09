module Evaluation

using ..CounterfactualExplanations
using DataFrames
using Statistics

export benchmark, evaluate, default_measures
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
        counterfactual_explanation::Union{
            CounterfactualExplanation,
            Vector{CounterfactualExplanation},
        };
        measure::Union{Function,Vector{Function}} = default_measures,
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
        evaluation.id = 1:nrow(evaluation)
        if pivot_longer
            evaluation = stack(evaluation, Not(:id))
        end
        select!(evaluation, :id, :)
    end

    return evaluation
end

function evaluate(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    kwargs...
)
    evaluations = []
    for (i, ce) in enumerate(counterfactual_explanations)
        evaluation = evaluate(ce; output_format=:DataFrame)
        evaluation.sample .= i
        evaluations = [evaluations..., evaluation]
    end
    evaluations = reduce(vcat, evaluations)
    select!(evaluations, :sample, :id, :)
    return evaluations

end

    include("benchmark.jl")

end