"""
    CounterfactualExplanations.parallelize(
        parallelizer::MPIParallelizer,
        f::typeof(CounterfactualExplanations.generate_counterfactual),
        args...;
        kwargs...,
    )

Parallelizes the `CounterfactualExplanations.generate_counterfactual` function using `MPI.jl`. This function is used to generate counterfactual explanations.
"""
function CounterfactualExplanations.parallelize(
    parallelizer::MPIParallelizer,
    f::typeof(CounterfactualExplanations.generate_counterfactual),
    args...;
    verbose::Bool=false,
    kwargs...,
)

    # Extract positional arguments:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)
    target = args[2]
    data = args[3]
    M = args[4]
    generator = args[5]

    # Split counterfactuals into groups of approximately equal size:
    x = split_obs(counterfactuals, parallelizer.n_proc)
    x = MPI.scatter(x, parallelizer.comm)

    # Split targets into groups of approximately equal size if necessary:
    if typeof(target) <: AbstractArray
        target = split_obs(target, parallelizer.n_proc)
        target = MPI.scatter(target, parallelizer.comm)
    end

    # Split models into groups of approximately equal size if necessary:
    if typeof(M) <: AbstractArray
        M = split_obs(M, parallelizer.n_proc)
        M = MPI.scatter(M, parallelizer.comm)
    end

    # Split generators into groups of approximately equal size if necessary:
    if typeof(generator) <: AbstractArray
        generator = split_obs(generator, parallelizer.n_proc)
        generator = MPI.scatter(generator, parallelizer.comm)
    end

    # Evaluate function:
    if !parallelizer.threaded
        if parallelizer.rank == 0 && verbose
            output = @showprogress desc = "Generating counterfactuals ..." broadcast(
                x, target, M, generator
            ) do x, target, M, generator
                with_logger(NullLogger()) do
                    f(x, target, data, M, generator; kwargs...)
                end
            end
        else
            output = with_logger(NullLogger()) do
                f.(x, target, data, M, generator; kwargs...)
            end
        end
    else
        second_parallelizer = ThreadsParallelizer()
        output = CounterfactualExplanations.parallelize(
            second_parallelizer, f, x, target, data, M, generator; kwargs...
        )
    end
    MPI.Barrier(parallelizer.comm)

    # Collect output from all processe in rank 0:
    collected_output = MPI.gather(output, parallelizer.comm)
    if parallelizer.rank == 0
        output = vcat(collected_output...)
    else
        output = nothing
    end
    MPI.Barrier(parallelizer.comm)

    # Broadcast output to all processes:
    final_output = MPI.bcast(output, parallelizer.comm; root=0)
    MPI.Barrier(parallelizer.comm)

    return final_output
end