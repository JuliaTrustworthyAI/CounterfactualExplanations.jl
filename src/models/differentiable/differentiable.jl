
"""
Base type for differentiable models.
"""
abstract type AbstractDifferentiableModel <: AbstractFittedModel end

"""
Base type for differentiable models written in pure Julia.
"""
abstract type AbstractDifferentiableJuliaModel <: AbstractDifferentiableModel end

include("flux-utils.jl")
include("flux-mlp.jl")
include("flux-ensemble.jl")
include("laplace-redux.jl")
