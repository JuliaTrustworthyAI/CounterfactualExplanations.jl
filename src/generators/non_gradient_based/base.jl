using Flux
using LinearAlgebra
using Parameters

abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

"Base class for heuristic/tree based counterfactual generators."
mutable struct HeuristicBasedGenerator <: AbstractNonGradientBasedGenerator
    penalty::Union{Nothing,Function,Vector{Function}}
    ϵ::Union{Nothing,AbstractFloat}
    latent_space::Bool
end

"""
    HeuristicBasedGenerator(;
        penalty::Union{Nothing,Function,Vector{Function}}=nothing,
        ϵ::Union{Nothing,AbstractFloat}=nothing,
    )

Default outer constructor for `HeuristicBasedGenerator`.
"""
function HeuristicBasedGenerator(;
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    ϵ::Union{Nothing,AbstractFloat}=nothing,
    latent_space::Bool=false,
)
    return HeuristicBasedGenerator(penalty, ϵ, latent_space)
end