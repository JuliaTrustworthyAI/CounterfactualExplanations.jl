"The `ThreadsParallelizer` type is used to parallelize the evaluation of a function using `Threads.@threads`."
struct ThreadsParallelizer <: CounterfactualExplanations.AbstractParallelizer end

"""
    CounterfactualExplanations.parallelize(
        parallelizer::ThreadsParallelizer,
        f::typeof(CounterfactualExplanations.generate_counterfactual),
        args...;
        kwargs...,
    )

Parallelizes the evaluation of `f` using `Threads.@threads`. This function is used to generate counterfactual explanations.
"""
function CounterfactualExplanations.parallelize(
    parallelizer::ThreadsParallelizer,
    f::typeof(CounterfactualExplanations.generate_counterfactual),
    args...;
    verbose::Bool=true,
    kwargs...,
)

    # Extract positional arguments:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)
    target = args[2]
    data = args[3]
    M = args[4]
    generator = args[5]

    @assert typeof(M) <: CounterfactualExplanations.AbstractFittedModel ||
        length(M) == length(counterfactuals) "The number of models must match the number of counterfactuals or be a single model."
    @assert typeof(generator) <: CounterfactualExplanations.AbstractGenerator ||
        length(generator) == length(counterfactuals) "The number of generators must match the number of counterfactuals or be a single generator."

    # Zip arguments (THIS CAN PROBABLY BE DONE BETTER):
    args = zip(
        counterfactuals,
        target |> x -> if typeof(target) <: AbstractArray
            target
        else
            fill(target, length(counterfactuals))
        end,
        fill(data, length(counterfactuals)),
        M |> x -> if typeof(M) <: CounterfactualExplanations.AbstractFittedModel
            fill(M, length(counterfactuals))
        else
            M
        end,
        generator |>
        x -> if typeof(generator) <: CounterfactualExplanations.AbstractGenerator
            fill(generator, length(counterfactuals))
        else
            generator
        end,
    )

    # Preallocate:
    ces = [
        Vector{CounterfactualExplanations.AbstractCounterfactualExplanation}() for
        _ in 1:Threads.nthreads()
    ]

    # Verbosity:
    if verbose
        prog = ProgressMeter.Progress(
            length(args);
            desc="Generating counterfactuals ...",
            showspeed=true,
            color=:green,
        )
    end

    # Training:  
    Threads.@threads for (x, target, data, M, generator) in collect(args)
        ce = with_logger(NullLogger()) do
            f(x, target, data, M, generator; kwargs...)
        end
        push!(ces[Threads.threadid()], ce)
        if verbose
            ProgressMeter.next!(prog)
        end
    end

    ces = reduce(vcat, ces)

    return ces
end

"""
    CounterfactualExplanations.parallelize(
        parallelizer::ThreadsParallelizer,
        f::typeof(CounterfactualExplanations.Evaluation.evaluate),
        args...;
        kwargs...,
    )

Parallelizes the evaluation of `f` using `Threads.@threads`. This function is used to evaluate counterfactual explanations.
"""
function CounterfactualExplanations.parallelize(
    parallelizer::ThreadsParallelizer,
    f::typeof(CounterfactualExplanations.Evaluation.evaluate),
    args...;
    verbose::Bool=true,
    kwargs...,
)

    # Setup:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)

    # Get meta data if supplied:
    if length(args) > 1
        meta_data = args[2]
    else
        meta_data = nothing
    end

    # Check meta data:
    if typeof(meta_data) <: AbstractArray
        meta_data = CounterfactualExplanations.vectorize_collection(meta_data)
        @assert length(meta_data) == length(counterfactuals) "The number of meta data must match the number of counterfactuals."
    else
        meta_data = fill(meta_data, length(counterfactuals))
    end

    # Preallocate:
    evaluations = [[] for _ in 1:Threads.nthreads()]

    # Verbosity:
    if verbose
        prog = ProgressMeter.Progress(
            length(counterfactuals);
            desc="Evaluating counterfactuals ...",
            showspeed=true,
            color=:green,
        )
    end

    Threads.@threads for i in eachindex(counterfactuals)
        push!(evaluations[Threads.threadid()], f(counterfactuals[i], meta_data[i]; kwargs...))
        if verbose
            ProgressMeter.next!(prog)
        end
    end
    evaluations = reduce(vcat, evaluations)

    return evaluations
end
