module GenerativeModels

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