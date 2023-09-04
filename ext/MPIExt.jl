module MPIExt

export MPIParallelizer

using CounterfactualExplanations
using MPI

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
end

"""
    MPIParallelizer(comm::MPI.Comm)

Create an `MPIParallelizer` object from an `MPI.Comm` object.
"""
function CounterfactualExplanations.MPIParallelizer(comm::MPI.Comm)
    rank = MPI.Comm_rank(comm)                          # Rank of this process in the world 🌍
    n_proc = MPI.Comm_size(comm)                        # Number of processes in the world 🌍

    if rank == 0
        @info "Using `MPI.jl` for multi-processing."
        println("Running on $n_proc processes.")
    end
    return MPIParallelizer(comm, rank, n_proc)
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
    output = f.(x, target, data, M, generator; kwargs...)

    MPI.Barrier(parallelizer.comm)

    output = MPI.gather(output, parallelizer.comm)

    # Collect output from all processe in rank 0:
    if parallelizer.rank == 0
        output = vcat(output...)
    else
        output = nothing
    end

    # Broadcast output to all processes:
    output = MPI.bcast(output, parallelizer.comm; root=0)

    MPI.Barrier(parallelizer.comm)

    return output
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
    kwargs...,
)

    # Setup:
    counterfactuals = args[1] |> x -> CounterfactualExplanations.vectorize_collection(x)

    # Split counterfactuals into groups of approximately equal size:
    x = split_obs(counterfactuals, parallelizer.n_proc)
    x = MPI.scatter(x, parallelizer.comm)

    # Split meta data into groups of approximately equal size:
    meta_data = args[2]
    if typeof(meta_data) <: AbstractArray
        meta_data = CounterfactualExplanations.vectorize_collection(meta_data)
        meta_data = split_obs(meta_data, parallelizer.n_proc)
        meta_data = MPI.scatter(meta_data, parallelizer.comm)
    end

    # Evaluate function:
    output = f.(x, meta_data; kwargs...)

    MPI.Barrier(parallelizer.comm)

    # Collect output from all processe in rank 0:
    output = MPI.gather(output, parallelizer.comm)
    if parallelizer.rank == 0
        output = vcat(output...)
    else
        output = nothing
    end

    # Broadcast output to all processes:
    output = MPI.bcast(output, parallelizer.comm; root=0)

    MPI.Barrier(parallelizer.comm)

    return output
end

end
