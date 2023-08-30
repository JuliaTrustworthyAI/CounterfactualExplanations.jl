module MPIExt

export MPIParallelizer

using CounterfactualExplanations
using MPI

include("utils.jl")

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
    collection = args[1] |> x -> vectorize_collection(x)
    if length(args) > 1
        _args = args[2:end]
    end

    chunks = split_obs(collection, parallelizer.n_proc)
    item = MPI.scatter(chunks, parallelizer.comm)
    if length(args) > 1
        output = f(item, _args...; kwargs...)
    else
        output = f(item; kwargs...)
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
