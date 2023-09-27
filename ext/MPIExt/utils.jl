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
