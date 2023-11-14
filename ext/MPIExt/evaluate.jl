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
    n_each = parallelizer.n_each

    # Extract positional arguments:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)
    # Get meta data if supplied:
    if length(args) > 1
        meta_data = args[2]
    else
        meta_data = nothing
    end
    meta_data =
        isa(meta_data, AbstractArray) ? meta_data : fill(meta_data, length(counterfactuals))

    # Break down into chunks:
    args = zip(counterfactuals, meta_data)
    if !isnothing(n_each)
        chunks = Parallelization.chunk_obs(args, n_each, parallelizer.n_proc)
    else
        chunks = [collect(args)]
    end

    # Setup:
    storage_path = tempdir()

    # For each chunk:
    for (i, chunk) in enumerate(chunks)
        worker_chunk = Parallelization.split_obs(chunk, parallelizer.n_proc)
        worker_chunk = MPI.scatter(worker_chunk, parallelizer.comm)
        worker_chunk = stack(worker_chunk; dims=1)
        if !parallelizer.threaded
            if parallelizer.rank == 0 && verbose
                # Generating counterfactuals with progress bar:
                output = []
                @showprogress desc = "Evaluating counterfactuals ..." for x in zip(
                    eachcol(worker_chunk)...
                )
                    with_logger(NullLogger()) do
                        push!(output, f(x...; kwargs...))
                    end
                end
            else
                # Generating counterfactuals without progress bar:
                output = with_logger(NullLogger()) do
                    f.(eachcol(worker_chunk)...; kwargs...)
                end
            end
        else
            # Parallelize further with `Threads.@threads`:
            second_parallelizer = ThreadsParallelizer()
            output = CounterfactualExplanations.parallelize(
                second_parallelizer, f, eachcol(worker_chunk)...; kwargs...
            )
        end
        MPI.Barrier(parallelizer.comm)

        # Collect output from all processe in rank 0:
        collected_output = MPI.gather(output, parallelizer.comm)
        if parallelizer.rank == 0
            output = vcat(collected_output...)
            Serialization.serialize(joinpath(storage_path, "output_$i.jls"), output)
        end
        MPI.Barrier(parallelizer.comm)
    end

    # Collect output from all processes on each process:
    outputs = []
    for i in 1:length(chunks)
        output = Serialization.deserialize(joinpath(storage_path, "output_$i.jls"))
        push!(outputs, output)
    end
    final_output = vcat(outputs...)
    MPI.Barrier(parallelizer.comm)

    return final_output
end
