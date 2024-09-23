"""
    AbstractNonGradientBasedGenerator

An abstract type that serves as the base type for non gradient-based counterfactual generators. 
"""
abstract type AbstractNonGradientBasedGenerator <: AbstractGenerator end

include("feature_tweak/feature_tweak.jl")
include("growing_spheres/growing_spheres.jl")
include("t_crex/t_crex.jl")
