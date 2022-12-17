module Models

using ..DataPreprocessing

export AbstractFittedModel, AbstractDifferentiableModel
export FluxModel, FluxEnsemble, LaplaceReduxModel, LogisticModel, BayesianLogisticModel
export probs, logits

"""
Base type for fitted models.
"""
abstract type AbstractFittedModel end

"""
    logits(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the raw model predictions. In classification this is sometimes referred to as *logits*: the non-normalized predictions that are fed into a link function to produce predicted probabilities. In regression (not currently implemented) raw outputs typically correspond to final outputs. In other words, there is typically no normalization involved.
"""
function logits(M::AbstractFittedModel, X::AbstractArray) end

"""
    probs(M::AbstractFittedModel, X::AbstractArray)

Generic method that is compulsory for all models. It returns the normalized model predictions, so the predicted probabilities in the case of classifiation. In regression (not currently implemented) this method is redundant. 
"""
function probs(M::AbstractFittedModel, X::AbstractArray) end

include("differentiable/differentiable.jl")
include("plotting.jl")

end
