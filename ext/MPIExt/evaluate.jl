"""
    parallelize(
        parallelizer::MPIParallelizer,
        f::typeof(CounterfactualExplanations.Evaluation.evaluate),
        args...;
        kwargs...,
    )

Parallelizes the evaluation of the `CounterfactualExplanations.Evaluation.evaluate` function. This function is used to evaluate the performance of a counterfactual explanation method. 
"""
function CounterfactualExplanations.parallelize(
    parallelizer::MPIParallelizer,
    f::typeof(CounterfactualExplanations.Evaluation.evaluate),
    args...;
    verbose::Bool=false,
    kwargs...,
)

    # Setup:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)

    # Split counterfactuals into groups of approximately equal size:
    x = split_obs(counterfactuals, parallelizer.n_proc)
    x = MPI.scatter(x, parallelizer.comm)

    # Get meta data if supplied:
    if length(args) > 1
        meta_data = args[2]
    else
        meta_data = nothing
    end

    # Split meta data into groups of approximately equal size:
    if typeof(meta_data) <: AbstractArray
        meta_data = CounterfactualExplanations.vectorize_collection(meta_data)
        meta_data = split_obs(meta_data, parallelizer.n_proc)
        meta_data = MPI.scatter(meta_data, parallelizer.comm)
    else
        meta_data = fill(meta_data, length(x))
    end

    # Evaluate function:
    if !parallelizer.threaded
        if parallelizer.rank == 0 && verbose
            output = @showprogress desc = "Evaluating counterfactuals ..." broadcast(
                x, meta_data
            ) do x, meta_data
                with_logger(NullLogger()) do
                    f(x, meta_data; kwargs...)
                end
            end
        else
            output = with_logger(NullLogger()) do
                f.(x, meta_data; kwargs...)
            end
        end
    else
        second_parallelizer = ThreadsParallelizer()
        output = CounterfactualExplanations.parallelize(
            second_parallelizer, f, meta_data; kwargs...
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