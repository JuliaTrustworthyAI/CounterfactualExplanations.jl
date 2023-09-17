module MPIExt

export MPIParallelizer

using CounterfactualExplanations
using CounterfactualExplanations.Parallelization
using Logging
using MPI
using ProgressMeter

### BEGIN utils.jl

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`. Lifted from https://juliaparallel.org/MPI.jl/v0.20/examples/06-scatterv/.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

"""
    split_by_counts(obs::AbstractVector, counts::AbstractVector)

Return a vector of vectors of `obs` split by `counts`.
"""
function split_by_counts(obs::AbstractVector, counts::AbstractVector)
    _start = cumsum([1; counts[1:(end - 1)]])
    _stop = cumsum(counts)
    return [obs[_start[i]:_stop[i]] for i in 1:length(counts)]
end

"""
    split_obs(obs::AbstractVector, n::Integer)

Return a vector of `n` group indices for `obs`.
"""
function split_obs(obs::AbstractVector, n::Integer)
    N = length(obs)
    N_counts = split_count(N, n)
    return split_by_counts(obs, N_counts)
end

### END 

"The `MPIParallelizer` type is used to parallelize the evaluation of a function using `MPI.jl`."
struct MPIParallelizer <: CounterfactualExplanations.AbstractParallelizer
    comm::MPI.Comm
    rank::Int
    n_proc::Int
    threaded::Bool
end

"""
    MPIParallelizer(comm::MPI.Comm)

Create an `MPIParallelizer` object from an `MPI.Comm` object.
"""
function CounterfactualExplanations.MPIParallelizer(comm::MPI.Comm; threaded::Bool=false)
    rank = MPI.Comm_rank(comm)                          # Rank of this process in the world 🌍
    n_proc = MPI.Comm_size(comm)                        # Number of processes in the world 🌍

    if rank == 0
        @info "Using `MPI.jl` for multi-processing."
        println("Running on $n_proc processes.")
    end
    return MPIParallelizer(comm, rank, n_proc, threaded)
end

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
            output = @showprogress desc="Generating counterfactuals ..." broadcast(x, target, M, generator) do x, target, M, generator
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
            second_parallelizer,
            f,
            x,
            target,
            data,
            M,
            generator;
            kwargs...,
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
            output = @showprogress desc="Evaluating counterfactuals ..." broadcast(x, meta_data) do x, meta_data
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
            second_parallelizer,
            f,
            meta_data;
            kwargs...,
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

end
