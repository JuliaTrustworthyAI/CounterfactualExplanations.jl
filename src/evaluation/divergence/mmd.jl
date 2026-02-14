# A lot of things salvaged from from IPMeasures: https://github.com/aicenter/IPMeasures.jl/blob/master/src/mmd.jl
using KernelFunctions
using Random

export MMD, mmd_null_dist, mmd_significance

include("kernels.jl")

"""
    MMD{K<:KernelFunctions.Kernel} <: AbstractDivergenceMetric

Concrete type for the Maximum Mean Discrepancy (MMD) metric.
"""
struct MMD{K<:KernelFunctions.Kernel} <: AbstractDivergenceMetric
    kernel::K
    compute_p::Union{Nothing,Int}
end

function MMD(; kernel=default_kernel, compute_p=1000)
    return MMD(kernel, compute_p)
end

CounterfactualExplanations.measure_name(m::MMD) = :mmd

"""
    (m::MMD)(x::AbstractArray, y::AbstractArray)

Computes the maximum mean discrepancy (MMD) between two datasets `x` and `y`. The MMD is a measure of the difference between two probability distributions. It is defined as the maximum value of the kernelized dot product between the two datasets. It is computed as the sum of average kernel values between columns (samples) of `x` and `y`, minus twice the average kernel value between columns (samples) of `x` and `y`. A larger MMD value indicates that the distributions are more different, while a value closer to zero suggests they are more similar. See also [`kernelsum`](@ref).
"""
function (m::MMD)(x::AbstractArray, y::AbstractArray)
    xx = kernelsum(m.kernel, x)
    yy = kernelsum(m.kernel, y)
    xy = kernelsum(m.kernel, x, y)
    mmd = xx + yy - 2xy
    if !isnothing(m.compute_p)
        mmd_null = mmd_null_dist(x, y, m.kernel; l=m.compute_p)
        p_val = mmd_significance(mmd, mmd_null)
    else
        p_val = NaN
    end
    return mmd, p_val
end

"""
    (m::MMD)(
        x::AbstractArray,
        y::AbstractArray,
        n::Int;
        kwrgs...
    )

Computes the MMD between two datasets `x` and `y`, along with a p-value based on a null distribution of MMD values (unless `m.compute_p=nothing`) for a random subset of the data (of sample size `n`). The p-value is computed using a permutation test.
"""
function (m::MMD)(
    x::AbstractArray,
    y::AbstractArray,
    n::Int;
    rng::AbstractRNG=Random.default_rng(),
    kwrgs...,
)
    return m(samplecolumns(rng, x, n), samplecolumns(rng, y, n); kwrgs...)
end

"""
    mmd_null_dist(
        x::AbstractArray, y::AbstractArray, k::KernelFunctions.Kernel=default_kernel; l=1000
    )

Compute the null distribution of MMD for two samples `x` and `y` through bootstrapping as follows:

1. For each bootstrap sample, shuffle the columns of `x` and `y`.
2. Compute the MMD between the shuffled samples.
3. Repeat this process `l` times to obtain a null distribution of MMD values.
4. Return the null distribution of MMD values.

Under the null hypothesis `x` and `y` are actually from the same distribution.
"""
function mmd_null_dist(
    x::AbstractArray, y::AbstractArray, k::KernelFunctions.Kernel=default_kernel; l=1000
)
    n = size(x, 2)
    mmd_null = zeros(l)
    Z = hcat(x, y)
    Zs = [Z[:, shuffle(1:end)] for i in 1:l]

    bootstrap = function (z)
        return MMD(k, nothing)(z[:, 1:n], z[:, (n + 1):end])[1]
    end

    mmd_null = map(Zs) do z
        res = bootstrap(z)
        return res
    end

    return mmd_null
end

"""
    mmd_significance(mmd::Number, mmd_null_dist::AbstractArray)

Compute the p-value of the MMD test as the proportion of MMD values in the null distribution that are greater than or equal to the observed MMD value.
"""
function mmd_significance(mmd::Number, mmd_null_dist::AbstractArray)
    return sum(mmd_null_dist .>= mmd) / length(mmd_null_dist)
end

"""
    samplecolumns([rng::AbstractRNG], x::AbstractMatrix, n::Int)

Sample `n` columns from a matrix. Returns `x` if the matrix has less than `n` columns.
"""
function samplecolumns(rng::AbstractRNG, x::AbstractMatrix, n::Int)
    return (size(x, 2) > n) ? x[:, sample(rng, 1:size(x, 2), n; replace=false)] : x
end

samplecolumns(x::AbstractMatrix, n::Int) = samplecolumns(Random.default_rng(), x, n)
