# --------------- Base type for model:
module Models

using NNlib, LinearAlgebra

include("base.jl")

export AbstractFittedModel, AbstractDifferentiableModel, 
    FluxModel, LogisticModel, BayesianLogisticModel,
    RTorchModel, PyTorchModel,
    probs, logits

end