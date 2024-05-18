
"Abstract types for differentiable models."
abstract type AbstractDifferentiableModelType <: AbstractModelType end

Differentiability(::AbstractDifferentiableModelType) = IsDifferentiable()

"""
Base type for differentiable models.
"""
abstract type AbstractDifferentiableModel <: AbstractModel end

Differentiability(::AbstractDifferentiableModel) = IsDifferentiable()

"""
Base type for differentiable models written in Flux.
"""
abstract type AbstractFluxModel <: AbstractDifferentiableModel end

"""
Base type for differentiable models from the MLJ library.
"""
abstract type AbstractMLJModel <: AbstractDifferentiableModel end

"""
Base type for custom differentiable models.
"""
abstract type AbstractCustomDifferentiableModel <: AbstractDifferentiableModel end

include("flux/flux.jl")
