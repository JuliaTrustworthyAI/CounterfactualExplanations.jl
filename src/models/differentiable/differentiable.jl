
"""
Base type for differentiable models.
"""
abstract type AbstractDifferentiableModel <: AbstractFittedModel end

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

"""
Base type for differentiable models written in Python.
"""
abstract type AbstractDifferentiablePythonModel <: AbstractDifferentiableModel end

include("flux/MLP.jl")
include("flux/ensemble.jl")
include("mlj/evo_tree.jl")
include("python/pytorch_model.jl")
include("other/laplace_redux.jl")
