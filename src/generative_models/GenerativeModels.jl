module GenerativeModels

using CounterfactualExplanations
using CUDA
using Flux
using Flux
using Flux.Losses
using Parameters
using ProgressMeter
using Random
using Statistics

# using CUDA
# using Flux
# using Flux: @functor, chunk, DataLoader
# using Flux.Losses: logitbinarycrossentropy, mse
# using Parameters: @with_kw
# using ProgressMeter
# using Random
# using Statistics

"""
Base type for generative model.
"""
abstract type AbstractGenerativeModel end

"""
Base type of generative model hyperparameter container.
"""
abstract type AbstractGMParams end

include("vae.jl")

end
