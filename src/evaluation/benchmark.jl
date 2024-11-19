using Base.Iterators
using DataFrames: DataFrames
using Serialization: Serialization
using Statistics: mean
using TaijaBase: AbstractParallelizer, vectorize_collection, parallelize
using UUIDs: UUIDs

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
        models::Dict{<:Any,<:AbstractModel},
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
    models::Dict{<:Any,<:AbstractModel},
    generators::Dict{<:Any,<:AbstractGenerator},
    measure::Union{Function,Vector{Function}}=default_measures,
    dataname::Union{Nothing,Symbol,String}=nothing,
    verbose::Bool=true,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    kwrgs...,
)
    xs = vectorize_collection(xs)

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
        parallelizer,
        generate_counterfactual,
        xs,
        target,
        data,
        Ms,
        gens;
        verbose=verbose,
        kwrgs...,
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
        test_data::Union{Nothing,CounterfactualData}=nothing,
        models::Dict{<:Any,<:Any}=standard_models_catalogue,
        generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
        measure::Union{Function,Vector{Function}}=default_measures,
        n_individuals::Int=5,
        n_runs::Int=1,
        suppress_training::Bool=false,
        factual::Union{Nothing,RawTargetType}=nothing,
        target::Union{Nothing,RawTargetType}=nothing,
        store_ce::Bool=false,
        parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
        dataname::Union{Nothing,Symbol,String}=nothing,
        verbose::Bool=true,
        vertical_splits::Union{Nothing,Int}=nothing,
        storage_path::String=tempdir(),
        kwrgs...,
    )

Benchmark a set of counterfactuals for a given data set and additional inputs.

# Arguments

- `data::CounterfactualData`: The dataset containing the factual and target labels.
- `test_data::Union{Nothing,CounterfactualData}`: Optional test data for evaluation. Defaults to `nothing`, in which case `data` is used for evaluation. 
- `models::Dict{<:Any,<:Any}`: A dictionary of model objects keyed by their names. Defaults to `standard_models_catalogue`.
- `generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}`: Optional dictionary of generator functions keyed by their names. Defaults to `nothing`, in which case the whole [`generator_catalogue`](@ref) is used. 
- `measure::Union{Function,Vector{Function}}`: The measure(s) to evaluate the counterfactuals against. Defaults to `default_measures`.
- `n_individuals::Int=5`: Number of individuals to generate for each model and generator. 
- `n_runs::Int=1`: Number of runs for each model and generator.
- `suppress_training::Bool=false`: Whether to suppress training of models during benchmarking. This is useful if models have already been trained.
- `factual::Union{Nothing,RawTargetType}`: Optional factual label. Defaults to `nothing`, in which case factual labels are randomly sampled from the dataset.
- `target::Union{Nothing,RawTargetType}`: Optional target label. Defaults to `nothing`, in which case target labels are randomly sampled from the dataset.
- `store_ce::Bool=false`: Whether to store the [`CounterfactualExplanation`](@ref) objects for each counterfactual.
- `parallelizer::Union{Nothing,AbstractParallelizer}=nothing`: Parallelization strategy for generating and evaluating counterfactuals.
- `dataname::Union{Nothing,Symbol,String}=nothing`: Name of the dataset. Defaults to `nothing`.
- `verbose::Bool=true`: Whether to print verbose output during benchmarking.
- `vertical_splits::Union{Nothing,Int}=nothing`: Number of vertical splits for generating counterfactuals. Defaults to `nothing`. This can useful, if it is necessary to reduce peak memory usage, by decreasing the number of counterfactuals generated at once. Higher values lead to smaller batches and hence smaller peak load, but lower degrees of parallelization.
- `storage_path::String=tempdir()`: Path where interim results will be stored. Defaults to `tempdir()`.
- `concatenate_output::Bool=true`: Whether to collect output from each run and concatenate them into a single output file.

# Returns

- A dictionary containing the benchmark results, including mean and standard deviation of the measures across all runs.

## Benchmarking Procedure

Runs the benchmarking exercise as follows:

1. Randomly choose a `factual` and `target` label unless specified. 
2. If no pretrained `models` are provided, it is assumed that a dictionary of callable model objects is provided (by default using the `standard_models_catalogue`). 
3. Each of these models is then trained on the data. 
4. For each model separately choose `n_individuals` randomly from the non-target (`factual`) class. For each generator create a benchmark as in [`benchmark(xs::Union{AbstractArray,Base.Iterators.Zip})`](@ref).
5. Finally, concatenate the results.

If `vertical_splits` is specified to an integer, the computations are split vertically into `vertical_splits` chunks. In this case, the results are stored in a temporary directory and concatenated afterwards. 
"""
function benchmark(
    data::CounterfactualData;
    test_data::Union{Nothing,CounterfactualData}=nothing,
    models::Dict{<:Any,<:Any}=standard_models_catalogue,
    generators::Union{Nothing,Dict{<:Any,<:AbstractGenerator}}=nothing,
    measure::Union{Function,Vector{Function}}=default_measures,
    n_individuals::Int=5,
    n_runs::Int=1,
    suppress_training::Bool=false,
    factual::Union{Nothing,RawTargetType}=nothing,
    target::Union{Nothing,RawTargetType}=nothing,
    store_ce::Bool=false,
    parallelizer::Union{Nothing,AbstractParallelizer}=nothing,
    dataname::Union{Nothing,Symbol,String}=nothing,
    verbose::Bool=true,
    vertical_splits::Union{Nothing,Int}=nothing,
    storage_path::String=tempdir(),
    concatenate_output::Bool=true,
    kwrgs...,
)

    # Setup:
    test_data = isnothing(test_data) ? data : test_data
    bmks = Benchmark[]
    split_vertically = !isnothing(vertical_splits)

    # Set up search:
    # If no factual is provided, choose randomly from the data for all individuals. Otherwise, use the same factual for all individuals.
    factual = if isnothing(factual)
        rand(data.y_levels, n_individuals)
    else
        fill(factual, n_individuals)
    end
    if isnothing(target)
        # If no target is provided, choose randomly from the data for all individuals, each time excluding the factual.
        target = [
            rand(data.y_levels[data.y_levels .!= factual[i]]) for i in 1:n_individuals
        ]
    else
        # Otherwise, use the same target for all individuals.
        target = fill(target, n_individuals)
    end

    # Train models if necessary:
    if !suppress_training
        @info "Training models on data."
        if typeof(models) <: Dict{<:Any,<:AbstractModel}
            models = Dict(key => Models.train(model, data) for (key, model) in models)
        else
            models = Dict(key => Models.fit_model(data, model()) for (key, model) in models)
        end
    end

    # Use all generators if none are provided:
    generators = if isnothing(generators)
        Dict(key => gen() for (key, gen) in generator_catalogue)
    else
        generators
    end

    # Run benchmarking exercise `n_runs` times:
    for run in 1:n_runs

        # General setup:
        if verbose
            @info "Run $run of $n_runs."
        end

        # Create a directory for this run:
        path_for_run = mkpath(joinpath(storage_path, "run_$run"))

        # Grid setup:
        grid = []
        for (mod_name, M) in models
            # Individuals need to be chosen separately for each model:
            chosen = Vector{Int}()
            yhat = CounterfactualExplanations.predict_label(M, test_data)
            for i in 1:n_individuals
                # For each individual and specified factual label, randomly choose index of a factual observation:
                chosen_ind = rand(
                    findall(
                        yhat .== factual[i],
                    ),
                )[1]
                push!(chosen, chosen_ind)
            end
            xs = CounterfactualExplanations.select_factual(test_data, chosen)
            xs = vectorize_collection(xs)
            # Form the grid:
            for (i, x) in enumerate(xs)
                sample_id = uuid1()
                for (gen_name, gen) in generators
                    comb = (x, target[i], (mod_name, M), (gen_name, gen), sample_id)
                    push!(grid, comb)
                end
            end
        end

        # Split grid vertically into `vertical_splits` parts:
        if split_vertically
            # Split grid vertically:
            grids = partition(grid, Int(ceil(length(grid) / vertical_splits)))
        else
            grids = [grid]
        end

        # For each grid:
        for (i, grid) in enumerate(grids)

            # Vectorize the grid:
            xs = [x[1] for x in grid]
            targets = [x[2] for x in grid]
            Ms = [x[3][2] for x in grid]
            gens = [x[4][2] for x in grid]
            sample_ids = [x[5] for x in grid]

            # Info:
            if split_vertically
                @info "Split $i of $(length(grids)) for run $run. Each grid has $(length(targets)) samples."
                id = get_global_output_id()
                output_path = joinpath(path_for_run, "output_$(i)_$(id).jls")
                if isfile(output_path)
                    # If final output is supposed to be concatenated, deserialize:
                    if concatenate_output && _serialization_state
                        bmk = Serialization.deserialize(output_path)
                        push!(bmks, bmk)
                    end
                    continue
                end
            end

            # Generate counterfactuals; in parallel if so specified
            ces = parallelize(
                parallelizer,
                generate_counterfactual,
                xs,
                targets,
                data,
                Ms,
                gens;
                verbose=verbose,
                kwrgs...,
            )

            # Meta Data:
            meta_data = map(eachindex(ces)) do i
                # Meta Data:
                _dict = Dict(
                    :model => grid[i][3][1],
                    :generator => grid[i][4][1],
                    :sample => sample_ids[i],
                )
                # Add dataname if supplied:
                if !isnothing(dataname)
                    _dict[:dataname] = dataname
                end
                if n_runs > 1
                    _dict[:run] = run
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
            if split_vertically && _serialization_state
                Serialization.serialize(output_path, bmk)
            end
            push!(bmks, bmk)
        end
    end

    # Collect and return output:
    if concatenate_output
        bmk = reduce(vcat, bmks)
        return bmk
    else
        return bmks
    end
end

"""
    concatenate_benchmarks(storage_path::String)

Concatenates all benchmarks stored in `storage_path` into a single benchmark.
"""
function concatenate_benchmarks(storage_path::String)
    if !_serialization_state
        return nothing
    end
    bmk_files = get_benchmark_files(storage_path)
    bmks = Serialization.deserialize.(bmk_files) 
    bmks = vcat(bmks...)
    return bmks
end

"""
    get_benchmark_files(storage_path::String)

Returns a list of all benchmark files stored in `storage_path`.
"""
function get_benchmark_files(storage_path::String)
     # No results:
    if length(storage_path) == 0
        @warn "No interim results found"
        return nothing
    end

    # Load from storage path:
    bmk_files = []
    for file in readdir(storage_path)
        # Get folders:
        run_folder = joinpath(storage_path, file)
        !isdir(run_folder) && continue
        !contains(splitpath(run_folder)[end], "run_") && continue

        # Get files:
        run_files = []
        for file in readdir(run_folder)
            if contains(splitpath(file)[end], "output_")
                fname = joinpath(run_folder, file)
                push!(run_files, fname)
            end
        end

        # Add to list:
        push!(bmk_files, run_files...)
    end

    return bmk_files
end
