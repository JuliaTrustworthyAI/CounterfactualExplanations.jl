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

    # Bundle arguments:
    args = zip(counterfactuals, meta_data)

    Threads.@threads :static for (ce, meta) in collect(args)
        push!(evaluations[Threads.threadid()], f(ce, meta; kwargs...))
        if verbose
            ProgressMeter.next!(prog)
        end
    end
    evaluations = reduce(vcat, evaluations)

    return evaluations
end
