using Flux
using LinearAlgebra
using Parameters

"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"Base class for counterfactual generators."
mutable struct Generator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}
    complexity::Union{Nothing,Function,Vector{Function}}
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}
    decision_threshold::Union{Nothing,AbstractFloat}
    latent_space::Bool
    opt::Flux.Optimise.AbstractOptimiser
    τ::AbstractFloat
end

"""
    Generator(;
        loss::Union{Nothing,Function}=nothing,
        complexity::Union{Nothing,Function,Vector{Function}}=nothing,
        λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing,
        decision_threshold::Union{Nothing,AbstractFloat}=nothing,
        latent_space::Bool::false,
    )

Default outer constructor for `Generator`.
"""
function Generator(;
    loss::Union{Nothing,Function}=nothing,
    complexity::Union{Nothing,Function,Vector{Function}}=nothing,
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}=nothing,
    decision_threshold::Union{Nothing,AbstractFloat}=nothing,
    latent_space::Bool=false,
    opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
    τ::AbstractFloat = CounterfactualExplanations.parameters[:τ],
)
    return Generator(loss, complexity, λ, decision_threshold, latent_space, opt, τ)
end


