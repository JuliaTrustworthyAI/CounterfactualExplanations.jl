using DataFrames
using Statistics
using ..Models: train

"A container for benchmarks of counterfactual explanations."
struct Benchmark
    counterfactual_explanations::Vector{CounterfactualExplanation}
    evaluation::DataFrame
end

"""
    (bmk::Benchmark)(; agg=mean)

Returns a `DataFrame` containing evaluation measures aggregated by `num_counterfactual`.
"""
function (bmk::Benchmark)(;agg::Union{Nothing,Function}=mean)
    df = bmk.evaluation
    if !isnothing(agg)
        df = combine(groupby(df, Not([:num_counterfactual, :value])), :value => agg => :value)
        select!(df, :sample, :variable, :value, :)
    end
    return df
end

"""
    Base.vcat(bmk1::Benchmark, bmk2::Benchmark)

Vertically concatenates two `Benchmark` objects.
"""
function Base.vcat(bmk1::Benchmark, bmk2::Benchmark; ids::Union{Nothing,AbstractArray}=nothing, idcol_name="dataset")
    @assert isnothing(ids) || length(ids) == 2
    ces = vcat(bmk1.counterfactual_explanations, bmk2.counterfactual_explanations)
    if !isnothing(ids)
        bmk1.evaluation[!,idcol_name] .= ids[1]
        bmk2.evaluation[!,idcol_name] .= ids[2]
    end
    evaluation = vcat(bmk1.evaluation, bmk2.evaluation)
    bmk = Benchmark(ces, evaluation)
    return bmk
end

function aggregate(bmk::Benchmark)

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
    evaluations = evaluate(
        counterfactual_explanations;
        measure=measure, report_each=true,
        report_meta=true, meta_data=meta_data
    )
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
    xids::Union{Nothing,AbstractArray}=nothing,
    kwrgs...
)

    @assert isnothing(xids) || length(xids) == length(x)

    # Counterfactual Search:
    meta_data = Vector{Dict}()
    ces = Vector{CounterfactualExplanation}()
    for (_sample, kv_pair) in enumerate(models)
        model_name = kv_pair[1]
        model = kv_pair[2]
        _sample = _sample * length(generators) - length(generators) + 1
        for (gen_name, generator) in generators
            _ces = generate_counterfactual(x, target, data, model, generator; kwrgs...)
            _ces = typeof(_ces) <: CounterfactualExplanation ? [_ces] : _ces
            push!(ces, _ces...)
            _meta_data = map(eachindex(_ces)) do i 
                sample_id = isnothing(xids) ? i : xids[i]
                Dict(:model => model_name, :generator => gen_name, :sample => sample_id) 
            end
            push!(meta_data, _meta_data...)
            _sample += 1
        end
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
    suppress_training::Bool=false,
    kwrgs...
)
    # Setup
    factual = rand(data.y_levels)
    target = rand(data.y_levels[data.y_levels.!=factual])
    if !suppress_training
        @info "Training models on data."
        if typeof(models) <: Dict{<:Any,<:AbstractFittedModel}
            models = Dict(key => train(model,data) for (key, model) in models)
        else
            models = Dict(key => train(model(data),data) for (key, model) in models)
        end
    end
    generators = isnothing(generators) ? Dict(key => gen() for (key, gen) in generator_catalogue) : generators

    # Performance Evaluation:
    bmk = Vector{Benchmark}()
    for (i, kv) in enumerate(models)
        key = kv[1]
        M = kv[2]
        chosen = rand(findall(predict_label(M, data) .== factual), n_individuals)
        xs = select_factual(data, chosen)
        _models = Dict(key => M)
        xids = (i - 1) * n_individuals .+ collect(1:n_individuals) # unique ids for samples
        _bmk = benchmark(xs, target, data; models=_models, generators=generators, measure=measure, xids=xids, kwrgs...)
        push!(bmk, _bmk)
    end
    bmk = reduce(vcat, bmk)

    return bmk
end