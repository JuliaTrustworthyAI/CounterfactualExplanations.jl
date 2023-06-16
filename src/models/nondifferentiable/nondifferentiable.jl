"""
Base type for non-differentiable models.
"""
abstract type AbstractNonDifferentiableModel <: AbstractFittedModel end

"""
Base type for non-differentiable models written in pure Julia.
"""
abstract type AbstractNonDifferentiableJuliaModel <: AbstractNonDifferentiableModel end

include("other/tree.jl")
