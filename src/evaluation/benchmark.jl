using DataFrames

struct Benchmark
    counterfactual_explanations::Vector{CounterfactualExplanation}
    evaluation::DataFrame
end

"""
    Base.vcat(bmk1::Benchmark, bmk2::Benchmark)

Vertically concatenates two `Benchmark` objects.
"""
function Base.vcat(bmk1::Benchmark, bmk2::Benchmark)
    ces = vcat(bmk1.counterfactual_explanations, bmk2.counterfactual_explanations)
    evaluation = vcat(bmk1.evaluation, bmk2.evaluation)
    bmk = Benchmark(ces, evaluation)
    return bmk
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

"""
    benchmark(
        x::Union{AbstractArray,Base.Iterators.Zip},
        target::RawTargetType,
        data::CounterfactualData;
        models::Dict{<:Any, <:AbstractFittedModel},
        generators::Dict{<:Any, <:AbstractGenerator},
        measure::Union{Function,Vector{Function}}=default_measures,
        kwrgs...
    )

First generates counterfactual explanations for factual `x`, the `target` and `data` using each of the provided `models` and `generators`. Then generates a `Benchmark` for the vector of counterfactual explanations as in [`benchmark(counterfactual_explanations::Vector{CounterfactualExplanation})`](@ref).
"""
function benchmark(
    x::Union{AbstractArray,Base.Iterators.Zip},
    target::RawTargetType,
    data::CounterfactualData;
    models::Dict{<:Any,<:AbstractFittedModel},
    generators::Dict{<:Any,<:AbstractGenerator},
    measure::Union{Function,Vector{Function}}=default_measures,
    kwrgs...
)
    # Counterfactual Search:
    meta_data = Vector{Dict}()
    ces = Vector{CounterfactualExplanation}()
    for (model_name, model) in models, (gen_name, generator) in generators
        _ces = generate_counterfactual(x, target, data, model, generator; kwrgs...)
        _ces = typeof(_ces) <: CounterfactualExplanation ? [_ces] : _ces
        push!(ces, _ces...)
        _meta_data = [Dict(:model => model_name, :generator => gen_name) for i in eachindex(_ces)]
        push!(meta_data, _meta_data...)
    end

    # Performance Evaluation:
    bmk = benchmark(ces; meta_data=meta_data, measure=measure)
    return bmk
end

"""
    benchmark(
        data::CounterfactualData;
        models::Dict{Symbol,Any}=model_catalogue,
        generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
        measure::Union{Function,Vector{Function}}=default_measures,
        n_individuals::Int=5,
        kwrgs...
    )

Runs the benchmarking exercise as follows:

1. Randomly choose a `factual` and `target` label. 
2. If no pretrained `models` are provided, it is assumed that a dictionary of callable model objects is provided (by default using the `model_catalogue`). 
3. Each of these models is then trained on the data. 
4. For each model separately choose `n_individuals` randomly from the non-target (`factual`) class. For each generator create a benchmark as in [`benchmark(x::Union{AbstractArray,Base.Iterators.Zip},...)`](@ref).
5. Finally, concatenate the results.

"""
function benchmark(
    data::CounterfactualData;
    models::Dict{Symbol,Any}=model_catalogue,
    generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures,
    n_individuals::Int=5,
    kwrgs...
)
    # Setup
    factual = rand(data.y_levels)
    target = rand(data.y_levels[data.y_levels.!=factual])
    if !(typeof(models) <: Dict{<:Any,<:AbstractFittedModel})
        @info "Training models on data."
        models = Dict(key => model(data) for (key, model) in models)
    end
    generators = isnothing(generators) ? Dict(key => gen() for (key, gen) in generator_catalog) : generators

    # Performance Evaluation:
    bmk = Vector{Benchmark}()
    for (key, M) in models
        chosen = rand(findall(predict_label(M, data) .== factual), n_individuals)
        x = select_factual(data, chosen)
        _models = Dict(key => M)
        _bmk = benchmark(x, target, data; models=_models, generators=generators, measure=measure, kwrgs...)
        push!(bmk, _bmk)
    end
    bmk = reduce(vcat, bmk)
    
    return bmk
end