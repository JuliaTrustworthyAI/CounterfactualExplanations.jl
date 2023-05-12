using Flux
using LinearAlgebra
using Parameters

abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

"Base class for heuristic/tree based counterfactual generators."
mutable struct HeuristicBasedGenerator <: AbstractNonGradientBasedGenerator
    loss::Union{Nothing,Function}
    ϵ::Union{Nothing,AbstractFloat}
end

"""
    HeuristicBasedGenerator(;
        loss::Union{Nothing,Function}=nothing,
        ϵ::Union{Nothing,AbstractFloat}=nothing,
    )

Default outer constructor for `HeuristicBasedGenerator`.
"""
function HeuristicBasedGenerator(;
    loss::Union{Nothing,Function}=nothing,
    ϵ::Union{Nothing,AbstractFloat}=nothing,
)
    return HeuristicBasedGenerator(loss, ϵ)
end