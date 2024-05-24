"""
    exp_cdf_transform(x::AbstractFloat, λ::AbstractFloat)

Computes the exponential cumulative distribution function (CDF) transformation of `x` with rate parameter `λ`.
"""
function exp_cdf_transform(x::AbstractFloat, λ::AbstractFloat=1.0)
    @assert λ > 0 "λ must be positive."
    return 1 - exp(-λ * x)
end

"""
    exp_decay(x::AbstractFloat, λ::AbstractFloat)

Computes the exponential decay of `x` with rate parameter `λ`.
"""
function exp_decay(x::AbstractFloat, λ::AbstractFloat=1.0)
    @assert λ > 0 "λ must be positive."
    return exp(-λ * x)
end
