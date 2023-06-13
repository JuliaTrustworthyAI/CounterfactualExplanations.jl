
"""
Base type for differentiable models.
"""
abstract type AbstractDifferentiableModel <: AbstractFittedModel end

"""
Base type for differentiable models written in pure Julia.
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

"""
Base type for models from the MLJ library.
"""
abstract type AbstractMLJModel <: AbstractDifferentiableModel end

include("flux_mlp.jl")
include("flux_ensemble.jl")
include("laplace_redux.jl")
include("evotree_model.jl")
