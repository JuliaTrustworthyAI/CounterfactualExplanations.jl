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



vectorize_collection(collection::Vector) = collection

vectorize_collection(collection::Base.Iterators.Zip) = map(x -> x[1], collect(collection))

function vectorize_collection(collection::Matrix)
    @warn "It looks like there is only one observation in the collection. Are you sure you want to parallelize?"
    return [collection]
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
    parallelize(
        parallelizer::MPIParallelizer,
        f::Function,
        args...;
        kwargs...,
    )

A function that can be used to multi-process the evaluation of `f`. The function `f` should be a function that takes a single argument. The argument should be a vector of counterfactual explanations. The function will split the vector of counterfactual explanations into groups of approximately equal size and distribute them to the processes. The results are then collected and returned.
"""
function CounterfactualExplanations.parallelize(
    parallelizer::MPIParallelizer, f::Function, args...; kwargs...
)
    @assert CounterfactualExplanations.parallelizable(f) "`f` is not a parallelizable process."

    # Setup:
    counterfactuals = args[1] |> x -> vectorize_collection(x)
    if length(args) > 1
        target = args[2]
        data = args[3]
        M = args[4]
        generator = args[5]
    end

    # Split counterfactuals into groups of approximately equal size:
    x = split_obs(counterfactuals, parallelizer.n_proc)
    x = MPI.scatter(x, parallelizer.comm)

    # Evaluate function:
    if length(args) > 1

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
    else
        output = f(x; kwargs...)
    end

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

end
