using Base.Iterators
using UUIDs

"A container for benchmarks of counterfactual explanations. Instead of subtyping `DataFrame`, it contains a `DataFrame` of evaluation measures (see [this discussion](https://discourse.julialang.org/t/creating-an-abstractdataframe-subtype/36451/6?u=pat-alt) for why we don't subtype `DataFrame` directly)."
struct Benchmark
    evaluation::DataFrames.DataFrame
end

"""
    (bmk::Benchmark)(; agg=mean)

Returns a `DataFrame` containing evaluation measures aggregated by `num_counterfactual`.
"""
function (bmk::Benchmark)(; agg::Union{Nothing,Function}=mean)
    df = bmk.evaluation
    if !isnothing(agg)
        df = DataFrames.combine(
            DataFrames.groupby(df, DataFrames.Not([:num_counterfactual, :value])),
            :value => agg => :value,
        )
        DataFrames.select!(df, :sample, :variable, :value, :)
    end
    return df
end

"""
    Base.vcat(bmk1::Benchmark, bmk2::Benchmark)

Vertically concatenates two `Benchmark` objects.
"""
function Base.vcat(
    bmk1::Benchmark,
    bmk2::Benchmark;
    ids::Union{Nothing,AbstractArray}=nothing,
    idcol_name="dataset",
)
    @assert isnothing(ids) || length(ids) == 2
    if !isnothing(ids)
        bmk1.evaluation[!, idcol_name] .= ids[1]
        bmk2.evaluation[!, idcol_name] .= ids[2]
    end
    evaluation = vcat(bmk1.evaluation, bmk2.evaluation)
    bmk = Benchmark(evaluation)
    return bmk
end

"""
    benchmark(
        counterfactual_explanations::Vector{CounterfactualExplanation};
        meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
        measure::Union{Function,Vector{Function}}=default_measures,
        store_ce::Bool=false,
    )

Generates a `Benchmark` for a vector of counterfactual explanations. Optionally `meta_data` describing each individual counterfactual explanation can be supplied. This should be a vector of dictionaries of the same length as the vector of counterfactuals. If no `meta_data` is supplied, it will be automatically inferred. All `measure` functions are applied to each counterfactual explanation. If `store_ce=true`, the counterfactual explanations are stored in the benchmark.
"""
function benchmark(
    counterfactual_explanations::Vector{CounterfactualExplanation};
    meta_data::Union{Nothing,<:Vector{<:Dict}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
)
    evaluations = parallelize(
        parallelizer,
        evaluate,
        counterfactual_explanations;
        measure=measure,
        report_each=true,
        report_meta=true,
        meta_data=meta_data,
        store_ce=store_ce,
    )
    bmk = Benchmark(evaluations)
    return bmk
end

"""
    benchmark(
        x::Union{AbstractArray,Base.Iterators.Zip},
        target::RawTargetType,
        data::CounterfactualData;
        models::Dict{<:Any,<:AbstractFittedModel},
        generators::Dict{<:Any,<:AbstractGenerator},
        measure::Union{Function,Vector{Function}}=default_measures,
        xids::Union{Nothing,AbstractArray}=nothing,
        dataname::Union{Nothing,Symbol,String}=nothing,
        verbose::Bool=true,
        store_ce::Bool=false,
        parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
        kwrgs...,
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
    dataname::Union{Nothing,Symbol,String}=nothing,
    verbose::Bool=true,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    kwrgs...,
)
    @assert isnothing(xids) || length(xids) == length(x)

    # Progress Bar:
    if verbose
        p_models = ProgressMeter.Progress(
            length(models); desc="Progress on models:", showspeed=true, color=:green
        )
        p_generators = ProgressMeter.Progress(
            length(generators); desc="Progress on generators:", showspeed=true, color=:blue
        )
    end

    # Counterfactual Search:
    meta_data = Vector{Dict}()
    ces = Vector{CounterfactualExplanation}()
    for (_sample, kv_pair) in enumerate(models)
        model_name = kv_pair[1]
        model = kv_pair[2]
        @info "Benchmarking model $model_name."
        _sample = _sample * length(generators) - length(generators) + 1
        for (gen_name, generator) in generators
            # Generate counterfactuals; in parallel if so specified
            _ces = parallelize(
                parallelizer,
                generate_counterfactual,
                x,
                target,
                data,
                model,
                generator;
                kwrgs...,
            )
            _ces = typeof(_ces) <: CounterfactualExplanation ? [_ces] : _ces
            push!(ces, _ces...)
            _meta_data = map(eachindex(_ces)) do i
                sample_id = isnothing(xids) ? uuid1() : xids[i]
                # Meta Data:
                _dict = Dict(
                    :model => model_name, :generator => gen_name, :sample => sample_id
                )
                # Add dataname if supplied:
                if !isnothing(dataname)
                    _dict[:dataname] = dataname
                end
                return _dict
            end
            push!(meta_data, _meta_data...)
            _sample += 1
            if verbose
                ProgressMeter.next!(
                    p_generators; showvalues=[(:model, model_name), (:generator, gen_name)]
                )
            end
        end
        if verbose
            ProgressMeter.next!(p_models)
        end
    end

    # Performance Evaluation:
    bmk = benchmark(
        ces;
        meta_data=meta_data,
        measure=measure,
        store_ce=store_ce,
        parallelizer=parallelizer,
    )
    return bmk
end

"""
    benchmark(
        data::CounterfactualData;
        models::Dict{<:Any,<:Any}=standard_models_catalogue,
        generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
        measure::Union{Function,Vector{Function}}=default_measures,
        n_individuals::Int=5,
        suppress_training::Bool=false,
        factual::Union{Nothing,RawTargetType}=nothing,
        target::Union{Nothing,RawTargetType}=nothing,
        store_ce::Bool=false,
        parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
        kwrgs...,
    )

Runs the benchmarking exercise as follows:

1. Randomly choose a `factual` and `target` label unless specified. 
2. If no pretrained `models` are provided, it is assumed that a dictionary of callable model objects is provided (by default using the `standard_models_catalogue`). 
3. Each of these models is then trained on the data. 
4. For each model separately choose `n_individuals` randomly from the non-target (`factual`) class. For each generator create a benchmark as in [`benchmark(x::Union{AbstractArray,Base.Iterators.Zip},...)`](@ref).
5. Finally, concatenate the results.

"""
function benchmark(
    data::CounterfactualData;
    models::Dict{<:Any,<:Any}=standard_models_catalogue,
    generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures,
    n_individuals::Int=5,
    suppress_training::Bool=false,
    factual::Union{Nothing,RawTargetType}=nothing,
    target::Union{Nothing,RawTargetType}=nothing,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    kwrgs...,
)
    # Setup
    factual = isnothing(factual) ? rand(data.y_levels) : factual
    target = isnothing(target) ? rand(data.y_levels[data.y_levels .!= factual]) : target
    if !suppress_training
        @info "Training models on data."
        if typeof(models) <: Dict{<:Any,<:AbstractFittedModel}
            models = Dict(key => Models.train(model, data) for (key, model) in models)
        else
            models = Dict(key => Models.train(model(data), data) for (key, model) in models)
        end
    end
    generators = if isnothing(generators)
        Dict(key => gen() for (key, gen) in generator_catalogue)
    else
        generators
    end

    # Grid setup:
    grid = []
    for M in values(models)
        # Individuals need to be chosen separately for each model:
        chosen = rand(
            findall(CounterfactualExplanations.predict_label(M, data) .== factual),
            n_individuals,
        )
        xs = CounterfactualExplanations.select_factual(data, chosen)
        # Form the grid:
        for x in xs
            for gen in generators
                comb = (x[1], M, gen)
                push!(grid, comb)
            end
        end
    end

    # Performance Evaluation:
    xs = [x[1] for x in grid]
    Ms = [x[2] for x in grid]
    gens = [x[3] for x in grid]

    ces = parallelize(
        parallelizer, 
        generate_counterfactual, 
        xs, target, data, Ms, gens; kwrgs...
    )

    bmk = Vector{Benchmark}()
    for (i, kv) in enumerate(models)
        key = kv[1]
        M = kv[2]
        chosen = rand(
            findall(CounterfactualExplanations.predict_label(M, data) .== factual),
            n_individuals,
        )
        xs = CounterfactualExplanations.select_factual(data, chosen)
        _models = Dict(key => M)

        xids = [uuid1() for x in 1:n_individuals]
        _bmk = benchmark(
            xs,
            target,
            data;
            models=_models,
            generators=generators,
            measure=measure,
            xids=xids,
            store_ce=store_ce,
            parallelizer=parallelizer,
            kwrgs...,
        )
        push!(bmk, _bmk)
    end
    bmk = reduce(vcat, bmk)

    return bmk
end
