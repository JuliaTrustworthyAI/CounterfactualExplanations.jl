using MPI

"""
    split_count(N::Integer, n::Integer)

Return a vector of `n` integers which are approximately equally sized and sum to `N`. Lifted from https://juliaparallel.org/MPI.jl/v0.20/examples/06-scatterv/.
"""
function split_count(N::Integer, n::Integer)
    q, r = divrem(N, n)
    return [i <= r ? q + 1 : q for i in 1:n]
end

"""
    split_obs(obs::AbstractVector, n::Integer)

Return a vector of `n` group indices for `obs`.
"""
function split_obs(obs::AbstractVector, n::Integer)
    N = length(obs)
    N_counts = split_count(N, n)
    _start = cumsum([1; N_counts[1:(end - 1)]])
    _stop = cumsum(N_counts)
    return [obs[_start[i]:_stop[i]] for i in 1:n]
end

"""
    with_mpi(expr)

A simple macro that can be used to specify the optimiser to be used.
"""
macro with_mpi(expr)
    idx = (expr.args .!= :evaluate) .&& (typeof.(expr.args) .!= Expr)
    ces = esc(expr.args[idx][1])
    evaluate_with_mpi = quote
        MPI.Init()

        comm = MPI.COMM_WORLD                               # Collection of processes that can communicate in our world 🌍
        rank = MPI.Comm_rank(comm)                          # Rank of this process in the world 🌍
        n_proc = MPI.Comm_size(comm)                        # Number of processes in the world 🌍

        chunks = split_obs($ces, n_proc)                    # Split ces into groups of approximately equal size
        ce = MPI.scatter(chunks, comm)                      # Scatter ces to all processes
        output = evaluate(ce)                               # Evaluate ces on each process

        MPI.Barrier(comm)                                   # Wait for all processes to reach this point

        # Collect output from all processes:
        if rank == 0
            output = MPI.gather(output, comm)               
            output = vcat(output...)
        end

        return output
    end
    return evaluate_with_mpi
end