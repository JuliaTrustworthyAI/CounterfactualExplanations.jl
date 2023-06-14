module GenerativeModels

using CounterfactualExplanations
using CUDA
using Flux
using Parameters
using ProgressMeter
using Random
using Statistics


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
