# --------------- Base type for model:
module Models

using NNlib, LinearAlgebra

include("functions.jl")

export AbstractFittedModel, AbstractDifferentiableModel, 
    LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

end