using Flux
using LinearAlgebra
using Parameters

abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

"Base class for heuristic/tree based counterfactual generators."
mutable struct HeuristicBasedGenerator <: AbstractNonGradientBasedGenerator
    loss::Union{Nothing,Function}
    penalty::Union{Nothing,Function,Vector{Function}}
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}
end

"""
    HeuristicBasedGenerator(;
        loss::Union{Nothing,Function}=nothing,
        penalty::Union{Nothing,Function,Vector{Function}}=nothing,
        λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing,
    )

Default outer constructor for `Generator`.
"""
function HeuristicBasedGenerator(;
    loss::Union{Nothing,Function}=nothing,
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}=nothing,
)
    return HeuristicBasedGenerator(loss, penalty, λ)
end