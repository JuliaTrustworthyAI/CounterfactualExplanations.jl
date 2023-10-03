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
    target = args[2] |> x -> isa(x, AbstractArray) ? x : fill(x, length(counterfactuals))
    data = args[3] |> x -> isa(x, AbstractArray) ? x : fill(x, length(counterfactuals))
    M = args[4] |> x -> isa(x, AbstractArray) ? x : fill(x, length(counterfactuals))
    generator = args[5] |> x -> isa(x, AbstractArray) ? x : fill(x, length(counterfactuals))

    # Break down into chunks:
    args = zip(counterfactuals, target, data, M, generator)

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
    Threads.@threads :static for (x, target, data, M, generator) in collect(args)
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
