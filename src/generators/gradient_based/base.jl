using Flux
using LinearAlgebra
using Parameters

"""
    AbstractGradientBasedGenerator

An abstract type that serves as the base type for gradient-based counterfactual generators. 
"""
abstract type AbstractGradientBasedGenerator <: AbstractGenerator end

"Base class for gradient-based counterfactual generators."
mutable struct GradientBasedGenerator <: AbstractGradientBasedGenerator
    loss::Union{Nothing,Function}
    penalty::Union{Nothing,Function,Vector{Function}}
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}
    latent_space::Bool
    opt::Flux.Optimise.AbstractOptimiser
end

"""
    Generator(;
        loss::Union{Nothing,Function}=nothing,
        penalty::Union{Nothing,Function,Vector{Function}}=nothing,
        λ::Union{Nothing,AbstractFloat,Vector{AbstractFloat}}=nothing,
        latent_space::Bool::false,
        opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
    )

Default outer constructor for `Generator`.
"""
function GradientBasedGenerator(;
    loss::Union{Nothing,Function}=nothing,
    penalty::Union{Nothing,Function,Vector{Function}}=nothing,
    λ::Union{Nothing,AbstractFloat,Vector{<:AbstractFloat}}=nothing,
    latent_space::Bool=false,
    opt::Flux.Optimise.AbstractOptimiser=Flux.Descent(),
)
    return GradientBasedGenerator(loss, penalty, λ, latent_space, opt)
end
