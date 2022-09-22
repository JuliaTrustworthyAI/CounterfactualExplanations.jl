################################################################################
# --------------- Base type for model:
################################################################################
"""
AbstractFittedModel

Base type for fitted models.
"""
abstract type AbstractFittedModel end

################################################################################
# --------------- Base type for differentiable model:
################################################################################
abstract type AbstractDifferentiableModel <: AbstractFittedModel end

include("differentiable/julia.jl") # Julia models
include("differentiable/R.jl") # R models
include("differentiable/Python.jl") # Python models