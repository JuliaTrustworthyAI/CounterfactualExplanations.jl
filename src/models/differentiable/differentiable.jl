
"""
Base type for differentiable models.
"""
abstract type AbstractDifferentiableModel <: AbstractFittedModel end

"""
Base type for differentiable models written in pure Flux.
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

include("flux/MLP.jl")
include("flux/ensemble.jl")
include("other/laplace_redux.jl")
include("other/evotree_model.jl")
