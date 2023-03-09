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

function benchmark(
    x::Union{AbstractArray,Base.Iterators.Zip},
    target::RawTargetType,
    data::CounterfactualData;
    models::Dict{<:Any, <:AbstractFittedModel},
    generators::Dict{<:Any, <:AbstractGenerator},
    measure::Union{Function,Vector{Function}}=default_measures,
    kwrgs...
)
    # Counterfactual Search:
    meta_data = Vector{Dict}()
    ces = Vector{CounterfactualExplanation}()
    for (model_name, model) in models, (gen_name, generator) in generators
        ce = generate_counterfactual(x,target,data,model,generator; kwrgs...)
        push!(ces, ce)
        push!(meta_data, Dict(:model => model_name, :generator => gen_name))
    end

    # Performance Evaluation:
    bmk = benchmark(ces; meta_data=meta_data, measure=measure)
    return bmk
end

# function benchmark(
#     data::CounterfactualData;
#     models::Union{Nothing,Dict{<:Any,<:AbstractFittedModel}},
#     generators::Dict{<:Any,<:AbstractGenerator},
#     measure::Union{Function,Vector{Function}}=default_measures,
#     n_individuals::Int=1,
#     kwrgs...
# )
#     # Setup
#     factual = rand(data.y_levels)
#     target = rand(data.y_levels[data.y_levels .!= factual])
#     chosen = rand(findall(predict_label(M, counterfactual_data) .== factual), n_individuals)
#     x = select_factual(counterfactual_data,chosen)

#     # Performance Evaluation:
#     bmk = benchmark(x, target, ces; models=models, generators=generators, measure=measure, kwrgs...)

#     return bmk
# end