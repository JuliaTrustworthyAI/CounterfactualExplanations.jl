"""
    compute_measure(ce::CounterfactualExplanation, measure::Function, agg::Function)

Computes a single measure for a counterfactual explanation. The measure is applied to the counterfactual explanation `ce` and aggregated using the aggregation function `agg`.
"""
function compute_measure(ce::CounterfactualExplanation, measure::Function, agg::Function)
    val = measure(ce; agg=agg)
    return ndims(val) > 1 ? vec(val) : [val]
end

"""
    evaluate_dict(ce::CounterfactualExplanation, measure::Vector{Function}, agg::Function)
Evaluates a counterfactual explanation and returns a dictionary of evaluation measures.
"""

function evaluate_dict(
    ce::CounterfactualExplanation, measure::Vector{Function}, agg::Function
)
    computed_measures = [compute_measure(ce, fun, agg) for fun in measure]
    return Dict(
        m => ndims(val) > 1 ? vec(val) : val for
        (m, val) in zip(Symbol.(measure), computed_measures)
    )
end

"""
    evaluate_dataframe(
        ce::CounterfactualExplanation,
        measure::Vector{Function},
        agg::Function,
        report_each::Bool,
        pivot_longer::Bool,
        store_ce::Bool,
    )
Evaluates a counterfactual explanation and returns a dataframe of evaluation measures.
"""
function evaluate_dataframe(
    ce::CounterfactualExplanation,
    measure::Vector{Function},
    agg::Function,
    report_each::Bool,
    pivot_longer::Bool,
    store_ce::Bool,
)
    computed_measures = [compute_measure(ce, fun, agg) for fun in measure]
    evaluation = DataFrames.DataFrame(
        Dict(
            m => report_each ? val[1] : val for
            (m, val) in zip(Symbol.(measure), computed_measures)
        ),
    )
    evaluation.num_counterfactual = 1:nrow(evaluation)
    if pivot_longer
        evaluation = DataFrames.stack(evaluation, DataFrames.Not(:num_counterfactual))
    end
    if store_ce
        evaluation.ce = repeat([ce], nrow(evaluation))
    end
    DataFrames.select!(evaluation, :num_counterfactual, :)
    return evaluation
end

"""
    generate_meta_data(i::Int, ce::CounterfactualExplanation, evaluation::DataFrame, report_meta::Bool, meta_data::Union{Nothing,Vector{Dict}})
Generates meta data for a counterfactual explanation. If `report_meta=true`, the meta data is extracted from the counterfactual explanation. If `meta_data` is supplied, it is used instead.
"""

function generate_meta_data(
    i::Int,
    ce::CounterfactualExplanation,
    evaluation::DataFrames.DataFrame,
    report_meta::Bool,
    meta_data::Union{Nothing,Vector{Dict}},
)
    if !isnothing(meta_data)
        df_meta = DataFrames.DataFrame(meta_data[i])
    else
        df_meta = DataFrames.DataFrame(CounterfactualExplanations.get_meta(ce))
    end
    if !("sample" ∈ names(df_meta))
        df_meta.sample .= i
    end
    evaluation = DataFrames.crossjoin(evaluation, df_meta; makeunique=true)
    evaluation.target .= ce.target
    evaluation.factual .= CounterfactualExplanations.factual_label(ce)
    return evaluation
end

"""
    evaluate(
        ce::CounterfactualExplanation;
        measure::Union{Function,Vector{Function}}=default_measures,
        agg::Function=mean,
        report_each::Bool=false,
        output_format::Symbol=:Vector,
        pivot_longer::Bool=true
    )

Just computes evaluation `measures` for the counterfactual explanation.
"""
function evaluate(
    ce::CounterfactualExplanation;
    measure::Union{Function,Vector{Function}}=default_measures,
    agg::Function=mean,
    report_each::Bool=false,
    output_format::Symbol=:Vector,
    pivot_longer::Bool=true,
    store_ce::Bool=false,
)
    @assert output_format ∈ [:Vector, :Dict, :DataFrame]
    measure = typeof(measure) <: Function ? [measure] : measure
    agg = report_each ? (x -> x) : agg
    if store_ce
        output_format = :DataFrame
    end

    if output_format == :Dict
        return evaluate_dict(ce, measure, agg)
    elseif output_format == :DataFrame
        return evaluate_dataframe(ce, measure, agg, report_each, pivot_longer, store_ce)
    else
        return [compute_measure(ce, fun, agg) for fun in measure]
    end
end

"""
    evaluate(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        report_meta::Bool=false,
        meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
        kwargs...
    )

Computes evaluation `measures` for a vector of counterfactual explanations. By default, no meta data is reported. For `report_meta=true`, meta data is automatically inferred, unless this overwritten by `meta_data`. The optional `meta_data` argument should be a vector of dictionaries of the same length as the vector of counterfactual explanations. 

Additional `kwargs...` can be provided (see [`evaluate(ce::CounterfactualExplanation`](@ref) for details).
"""
function evaluate(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    report_meta::Bool=false,
    meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
    kwargs...,
)
    if :output_format ∈ keys(kwargs)
        output_format = kwargs[:output_format]
        @assert output_format == :DataFrame ArgumentError(
            "Only output_format=:DataFrame supported for multiple counterfactual explanations",
        )
    end
    evaluations = []
    for (i, ce) in enumerate(counterfactual_explanations)
        evaluation = evaluate(ce; output_format=:DataFrame, kwargs...)
        if report_meta || !isnothing(meta_data)
            evaluation = generate_meta_data(i, ce, evaluation, report_meta, meta_data)
        end
        if !("sample" ∈ names(evaluation))
            evaluation.sample .= i
        end
        evaluations = [evaluations..., evaluation]
    end
    evaluations = reduce(vcat, evaluations)
    DataFrames.select!(evaluations, :sample, :num_counterfactual, :)
    return evaluations
end
