using UUIDs

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

function to_dict(computed_measures::Vector, measure)
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
function to_dataframe(
    computed_measures::Vector,
    measure,
    report_each::Bool,
    pivot_longer::Bool,
    store_ce::Bool,
)
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
    ce::CounterfactualExplanation,
    evaluation::DataFrames.DataFrame,
    meta_data::Union{Nothing,Vector{<:Dict}},
)
    if !isnothing(meta_data)
        df_meta = DataFrames.DataFrame(meta_data)
    else
        df_meta = DataFrames.DataFrame(CounterfactualExplanations.get_meta(ce))
    end
    evaluation = DataFrames.crossjoin(evaluation, df_meta; makeunique=true)
    evaluation[!, :target] .= ce.target
    evaluation[!, :factual] .= CounterfactualExplanations.factual_label(ce)
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

Just computes evaluation `measures` for the counterfactual explanation. By default, no meta data is reported. For `report_meta=true`, meta data is automatically inferred, unless this overwritten by `meta_data`. The optional `meta_data` argument should be a vector of dictionaries of the same length as the vector of counterfactual explanations. 
"""
function evaluate(
    ce::CounterfactualExplanation;
    measure::Union{Function,Vector{Function}}=default_measures,
    agg::Function=mean,
    report_each::Bool=false,
    output_format::Symbol=:Vector,
    pivot_longer::Bool=true,
    store_ce::Bool=false,
    report_meta::Bool=false,
    meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
)
    @assert output_format ∈ [:Vector, :Dict, :DataFrame]
    measure = typeof(measure) <: Function ? [measure] : measure
    agg = report_each ? (x -> x) : agg
    computed_measures = [compute_measure(ce, fun, agg) for fun in measure]

    if store_ce
        output_format = :DataFrame
    end

    if output_format == :Dict
        return to_dict(computed_measures, measure)
    elseif output_format == :DataFrame
        df = to_dataframe(computed_measures, measure, report_each, pivot_longer, store_ce)
        if report_meta || !isnothing(meta_data)
            df = generate_meta_data(ce, df, meta_data)
        end
        if !("sample" ∈ names(df))
            df.sample .= uuid1()
        end
        DataFrames.select!(df, :sample, :num_counterfactual, :)
        return df
    else
        return computed_measures
    end
end
