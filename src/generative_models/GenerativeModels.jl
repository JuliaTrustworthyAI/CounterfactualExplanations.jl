module GenerativeModels

using CounterfactualExplanations
using Flux
using ProgressMeter: ProgressMeter
using Random
using Statistics: Statistics

"""
Base type for generative model.
"""
abstract type AbstractGenerativeModel end

"""
Base type of generative model hyperparameter container.
"""
abstract type AbstractGMParams end

include("encoders.jl")
include("vae.jl")

end
