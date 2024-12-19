using DataFrames: nrow
using UUIDs: uuid1

"An abstract type for CE transformers."
abstract type AbstractCETransformer end

"Default CE transformer that returns the input as is."
struct IdentityTransformer <: AbstractCETransformer end

"Transformation function for default transformer."
TransformationFunction(transformer::IdentityTransformer) = (x -> x)

"Global CE transformer."
global _ce_transform = TransformationFunction(IdentityTransformer())

"Get the global CE transformer."
get_global_ce_transform() = _ce_transform

"The `ExplicitCETransformer` can be used to specify any arbitrary CE transformation."
struct ExplicitCETransformer <: AbstractCETransformer
    fun::Function
    function ExplicitCETransformer(fun::Function)
        @assert hasmethod(fun, Tuple{CounterfactualExplanation}) "Measure function must have a method for `CounterfactualExplanation`"
        return new(fun)
    end
end

"Transformation function for explicit transformer."
TransformationFunction(transformer::ExplicitCETransformer) = transformer.fun

"""
    global_ce_transform(transformer::AbstractCETransformer)

Sets the global CE transformer to `transformer`.
"""
function global_ce_transform(transformer::AbstractCETransformer)
    global _ce_transform = TransformationFunction(transformer)
    return _ce_transform
end

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
    to_dataframe(
        computed_measures::Vector,
        measure,
        report_each::Bool,
        pivot_longer::Bool,
        store_ce::Bool,
        ce::CounterfactualExplanation,
    )

Evaluates a counterfactual explanation and returns a dataframe of evaluation measures.
"""
function to_dataframe(
    computed_measures::Vector,
    measure,
    report_each::Bool,
    pivot_longer::Bool,
    store_ce::Bool,
    ce::CounterfactualExplanation,
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
        transform_fun = get_global_ce_transform()
        evaluation.ce = repeat([transform_fun(ce)], nrow(evaluation))
    end
    DataFrames.select!(evaluation, :num_counterfactual, :)
    return evaluation
end

"""
    generate_meta_data(
        ce::CounterfactualExplanation,
        evaluation::DataFrames.DataFrame,
        meta_data::Union{Nothing,Dict},
    )

Generates meta data for a counterfactual explanation. If `report_meta=true`, the meta data is extracted from the counterfactual explanation. If `meta_data` is supplied, it is used instead.
"""

function generate_meta_data(
    ce::CounterfactualExplanation,
    evaluation::DataFrames.DataFrame,
    meta_data::Union{Nothing,Dict},
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
        ce::CounterfactualExplanation,
        meta_data::Union{Nothing,Dict}=nothing;
        measure::Union{Function,Vector{Function}}=default_measures,
        agg::Function=mean,
        report_each::Bool=false,
        output_format::Symbol=:Vector,
        pivot_longer::Bool=true,
        store_ce::Bool=false,
        report_meta::Bool=false,
    )

Just computes evaluation `measures` for the counterfactual explanation. By default, no meta data is reported. For `report_meta=true`, meta data is automatically inferred, unless this overwritten by `meta_data`. The optional `meta_data` argument should be a vector of dictionaries of the same length as the vector of counterfactual explanations. 

## Arguments:

- `ce`: The counterfactual explanation to evaluate.
- `meta_data`: A vector of dictionaries containing meta data for each counterfactual explanation. If not provided, the default meta data is inferred from the counterfactual explanations.
- `measure`: The evaluation measures to compute. By default, all available measures are computed.
- `agg`: The aggregation function to use for the evaluation measures. By default, the mean is used.
- `report_each`: If true, each evaluation measure is reported separately. Otherwise, the mean of all measures is reported.
- `output_format`: The format of the output. By default, a vector is returned.
- `pivot_longer`: If true, the evaluation measures are pivoted longer. Otherwise, they are stacked.
- `store_ce`: If true, the counterfactual explanation is stored in the evaluation DataFrame. **Note**: These objects are potentially large and can consume a lot of memory.
- `report_meta`: If true, meta data is reported. Otherwise, it is not.

"""
function evaluate(
    ce::CounterfactualExplanation,
    meta_data::Union{Nothing,Dict}=nothing;
    measure::Union{Function,Vector{Function}}=default_measures,
    agg::Function=mean,
    report_each::Bool=false,
    output_format::Symbol=:Vector,
    pivot_longer::Bool=true,
    store_ce::Bool=false,
    report_meta::Bool=false,
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
        df = to_dataframe(
            computed_measures, measure, report_each, pivot_longer, store_ce, ce
        )
        if report_meta || !isnothing(meta_data)
            df = generate_meta_data(ce, df, meta_data)
        end
        if !("sample" ∈ names(df))
            df[!, "sample"] .= uuid1()
        end
        DataFrames.select!(df, :sample, :num_counterfactual, :)
        return df
    else
        return computed_measures
    end
end
