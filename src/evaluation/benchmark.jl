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
        counterfactual_explanations,
        meta_data;
        measure=measure,
        report_each=true,
        report_meta=true,
        store_ce=store_ce,
        output_format=:DataFrame,
    )
    bmk = Benchmark(reduce(vcat, evaluations))
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
    xs::Union{AbstractArray,Base.Iterators.Zip},
    target::RawTargetType,
    data::CounterfactualData;
    models::Dict{<:Any,<:AbstractFittedModel},
    generators::Dict{<:Any,<:AbstractGenerator},
    measure::Union{Function,Vector{Function}}=default_measures,
    dataname::Union{Nothing,Symbol,String}=nothing,
    verbose::Bool=true,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    kwrgs...,
)
    xs = CounterfactualExplanations.vectorize_collection(xs)

    # Grid setup:
    grid = []
    for (mod_name, M) in models
        for x in xs
            for (gen_name, gen) in generators
                comb = (x, (mod_name, M), (gen_name, gen))
                push!(grid, comb)
            end
        end
    end

    # Vectorize the grid:
    xs = [x[1] for x in grid]
    Ms = [x[2][2] for x in grid]
    gens = [x[3][2] for x in grid]

    # Generate counterfactuals; in parallel if so specified
    ces = parallelize(
        parallelizer, generate_counterfactual, xs, target, data, Ms, gens; verbose=verbose, kwrgs...
    )

    # Meta Data:
    meta_data = map(eachindex(ces)) do i
        sample_id = uuid1()
        # Meta Data:
        _dict = Dict(:model => grid[i][2], :generator => grid[i][3], :sample => sample_id)
        # Add dataname if supplied:
        if !isnothing(dataname)
            _dict[:dataname] = dataname
        end
        return _dict
    end

    # Evaluate counterfactuals; in parallel if so specified
    evaluations = parallelize(
        parallelizer,
        evaluate,
        ces,
        meta_data;
        measure=measure,
        report_each=true,
        report_meta=true,
        store_ce=store_ce,
        output_format=:DataFrame,
        verbose=verbose,
    )

    bmk = Benchmark(reduce(vcat, evaluations))

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
    dataname::Union{Nothing,Symbol,String}=nothing,
    verbose::Bool=true,
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
    for (mod_name, M) in models
        # Individuals need to be chosen separately for each model:
        chosen = rand(
            findall(CounterfactualExplanations.predict_label(M, data) .== factual),
            n_individuals,
        )
        xs = CounterfactualExplanations.select_factual(data, chosen)
        xs = CounterfactualExplanations.vectorize_collection(xs)
        # Form the grid:
        for x in xs
            for (gen_name, gen) in generators
                comb = (x, (mod_name, M), (gen_name, gen))
                push!(grid, comb)
            end
        end
    end

    # Vectorize the grid:
    xs = [x[1] for x in grid]
    Ms = [x[2][2] for x in grid]
    gens = [x[3][2] for x in grid]

    # Generate counterfactuals; in parallel if so specified
    ces = parallelize(
        parallelizer, generate_counterfactual, xs, target, data, Ms, gens; verbose=verbose, kwrgs...
    )

    # Meta Data:
    meta_data = map(eachindex(ces)) do i
        sample_id = uuid1()
        # Meta Data:
        _dict = Dict(
            :model => grid[i][2][1], :generator => grid[i][3][1], :sample => sample_id
        )
        # Add dataname if supplied:
        if !isnothing(dataname)
            _dict[:dataname] = dataname
        end
        return _dict
    end

    # Evaluate counterfactuals; in parallel if so specified
    evaluations = parallelize(
        parallelizer,
        evaluate,
        ces,
        meta_data;
        verbose=verbose,
        measure=measure,
        report_each=true,
        report_meta=true,
        store_ce=store_ce,
        output_format=:DataFrame,
    )
    bmk = Benchmark(reduce(vcat, evaluations))

    return bmk
end
