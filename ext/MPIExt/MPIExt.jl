module MPIExt

export MPIParallelizer

using CounterfactualExplanations
using CounterfactualExplanations.Parallelization
using Logging
using MPI
using ProgressMeter

include("utils.jl")

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

include("generate_counterfactual.jl")
include("evaluate.jl")

end
