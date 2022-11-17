using StatsBase

"""
    StatsBase.reconstruct(dt::ZScoreTransform, y::AbstractMatrix{<:Real}, idx::AbstractArray)

Method extension to avoid array mutation issue with Zygote.jl.
"""
function StatsBase.reconstruct(dt::ZScoreTransform, y::AbstractMatrix{<:Real}, idx::AbstractArray)
    D = size(y,1)
    _scale = ones(D)
    _scale[idx] .= dt.scale
    _mean = zeros(D)
    _mean[idx] .= dt.mean
    y = y .* _scale .+ _mean
    return y
end
