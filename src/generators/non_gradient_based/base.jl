using Flux
using LinearAlgebra
using Parameters

abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

"Base class for heuristic/tree based counterfactual generators."
mutable struct HeuristicBasedGenerator <: AbstractNonGradientBasedGenerator
    penalty::Union{Nothing,Function,Vector{Function}}
    ϵ::Union{Nothing,AbstractFloat}
end

"""
    HeuristicBasedGenerator(;
        enalty::Union{Nothing,Function,Vector{Function}}=nothing,
        ϵ::Union{Nothing,AbstractFloat}=nothing,
    )

Default outer constructor for `HeuristicBasedGenerator`.
"""
function HeuristicBasedGenerator(;
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    ϵ::Union{Nothing,AbstractFloat}=nothing,
)
    return HeuristicBasedGenerator(penalty, ϵ)
end